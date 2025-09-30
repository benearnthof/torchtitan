"""
llama3 style flexible transformer implementation
Adapted from
https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/model/model.py
Torchtitan requires the nightly PyTorch versions as context parallelism relies on some experimental features.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from torchtitan.models.attention import build_attention
from torchtitan.protocols.train_spec import ModelProtocol

from args import ImageTransformerArgs

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """repeat_interleave for GQA"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    Multi-head attention with GQA/MQA flexibility. To recover vanilla MHA set n_kv_heads to `None`.

    Args:
        model_args (ImageTransformerArgs): Model configuration arguments.
    
    Attributes: 
        n_kv_heads (int): Number of key and value heads.
        n_heads (int): Number of query heads.
        n_rep (int): Number of repetitions for local heads.
        head_dim (int): Dimension of each attention head.
        wq (nn.Linear): Linear layer for queries.
        wk (nn.Linear): Linear layer for keys.
        wv (nn.Linear): Linear layer for values.
        wo (nn.Linear): Linear layer for output.
    """

    def __init__(self, model_args: ImageTransformerArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        # we ditch half-size qk_dim since we now use GQA
        self.head_dim = model_args.dim // model_args.n_heads
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)
        # torch spda already applies dropout for us
        # torchtitan wraps this with device specific backend, DeepSpeed already does this for us (?)
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            # adjusted to be consistent with Sparse Transformers paper
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.125 / math.sqrt(linear.in_features))
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std) # for depth dependent init


    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor=None): # TODO: do we need to pass in freqs_cis?
        _ = freqs_cis
        bs, seqlen, _ = x.shape 
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # infer head size from -1 as TP may have sharded them after linear ops
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        # TODO: try RoPE instead of position embedding on images
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = keys.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xv = values.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)
        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    """
    FeedForward module
    Implements Gated Linear Units: https://arxiv.org/pdf/1612.08083
    Specifically, the SwiGLU variant proposed in: https://arxiv.org/pdf/2002.05202

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension of the feedforward layer.
        multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
        ffn_dim_multiplier (float | None): Custom multiplier for hidden dimension. Defaults to None.

    Attributes:
        w1 (Linear): Linear transformation for the first layer.
        w2 (Linear): Linear transformation for the second layer.
        w3 (Linear): Linear transformation for the third layer.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None
        ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.125 / math.sqrt(self.w1.in_features))
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std) # depth dependent init
    


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module
    Instead of the custom LayerNorm we switch over to RMSNorm https://arxiv.org/pdf/1910.07467

    Args:
        layer_id (int): Identifier for the layer, for layer-wise init.
        model_args (ImageTransformerArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output. 
        ffn_norm (RMSNorm): Layer normalization for feedforward output.
    """

    def __init__(self, layer_id: int, model_args: ImageTransformerArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)
        self.feed_forward = FeedForward(
            dim=model_args.dim,
            hidden_dim=model_args.dim,
            multiple_of=model_args.multiple_of,
            ffn_dim_multiplier=model_args.ffn_dim_multiplier,
        )
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor=None):
            """
            Perform a forward pass through the TransformerBlock.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: Output tensor after applying attention and feedforward layers.
            """
            # prenorm implementation
            h = x + self.attention(self.attention_norm(x), freqs_cis)
            out = h + self.feed_forward(self.ffn_norm(h))
            return out

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        self.feed_forward.init_weights(self.weight_init_std)


class Transformer(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.register_buffer(# needed for weight init and passthrough
    #         "freqs_cis", torch.zeros([1]), persistent=False
    #     )
    """
    Transformer Module

    Args:
        model_args (ImageTransformerArgs): Model configuration arguments.

    Attributes:
        model_args (ImageTransformerArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_emb (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
    """

    def __init__(self, model_args: ImageTransformerArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_emb = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.row_emb = nn.Embedding(32, model_args.dim)  # 32 rows
        self.col_emb = nn.Embedding(32, model_args.dim)  # 32 cols
        self.chan_emb = nn.Embedding(3, model_args.dim)  # 3 channels (RGB)

        self.register_buffer(# needed for weight init and passthrough
            "freqs_cis", torch.zeros([1]), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        def _init_embedding(module):
            n = module.num_embeddings
            d = module.embedding_dim
            # vocab embeddings are size 256, positional embeddings are size 32
            if n > 32: # vocab embedding # TODO: set to 64 for imagenet?
                std = 0.125 / math.sqrt(d)
            else: # row/col/chan embeddings
                n_emb = 3
                std = 0.125 / math.sqrt(d * n_emb)
            nn.init.normal_(module.weight, mean=0.0, std=std)

        if self.tok_emb is not None:
            # if stage has tok_emb it also has other embedding steps
            _init_embedding(self.tok_emb)
            _init_embedding(self.row_emb)
            _init_embedding(self.col_emb)
            _init_embedding(self.chan_emb)
        
        # already performs custom init with truncated normal
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        # final output layer is zero init (End of section 6 in the paper)
        if self.output is not None:
            torch.nn.init.zeros_(self.output.weight)
        

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
        ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # bypass in the case of pipeline parallelism
        if self.tok_emb:
            device = self.freqs_cis.device
            b, t = tokens.size()
            tok_emb = self.tok_emb(tokens)
            H, W, C = 32, 32, 3
            positions = torch.arange(t, device=device)
            chans = positions // (H * W)                 # 0..2
            rows  = (positions % (H * W)) // W           # 0..31
            cols  = positions % W              
            row_emb = self.row_emb(rows)[None, :, :].expand(b, -1, -1)
            col_emb = self.col_emb(cols)[None, :, :].expand(b, -1, -1)
            chan_emb = self.chan_emb(chans)[None, :, :].expand(b, -1, -1)

            h = (tok_emb + row_emb + col_emb + chan_emb)
        else:
            h = tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
        # logits.shape torch.Size([4, 3072, 256])
        # if targets is not None:
        #     loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
        #     loss = None
        # return logits, loss
        # TODO: loss is defined like so https://github.com/pytorch/torchtitan/blob/5b5d46856b400c8550989415bee91473aab4f921/torchtitan/components/loss.py#L19
        # this should be equivalent, but check nonetheless