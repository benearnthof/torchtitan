# Notes About parallelize.py

## World Mesh

## Tensor Parallelism

How it is implemented under the hood: https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487/7  

TorchTitan uses the `parallelize_module` api from torch.distributed for tensor parallelism. `apply_tp` is a loose wrapper around this. We simply pass in the transformer we wish to parallelize, the tp_mesh generated automatically by the training job and speficy the Parallelize plan for all embedding layers individually. If they are not in the respective module, like the "xd_emb" specified below, the respective plan just gets skipped.  
Transformer layers are then parallelized in a separate loop. 

```python
import torch
from torch import nn

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from torch.distributed.tensor import Replicate, Shard
from torch.distributed.device_mesh import init_device_mesh

tp_mesh = init_device_mesh("cuda", (1,))

def apply_tp(
    model: nn.Module,
    tp_mesh, # world_mesh["tp"] = parallel_dims.world_mesh is passed to TrainSpec
):
    """Apply tensor parallelism to the image transformer."""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_emb": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "row_emb": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "col_emb": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "chan_emb": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "xd_emb": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
        },
    )
    return model


class ImageEmbeddings(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.row_emb = nn.Embedding(32, dim)
        self.col_emb = nn.Embedding(32, dim)
        self.chan_emb = nn.Embedding(3, dim)

    def forward(self, tokens, device):
        b, t = tokens.size()
        tok_emb = self.tok_emb(tokens)
        H, W, C = 32, 32, 3
        positions = torch.arange(t, device=device)
        chans = positions // (H * W)
        rows  = (positions % (H * W)) // W
        cols  = positions % W
        row_emb  = self.row_emb(rows)[None, :, :].expand(b, -1, -1)
        col_emb  = self.col_emb(cols)[None, :, :].expand(b, -1, -1)
        chan_emb = self.chan_emb(chans)[None, :, :].expand(b, -1, -1)
        return tok_emb + row_emb + col_emb + chan_emb


model = ImageEmbeddings(256, 128)

out = apply_tp(model, tp_mesh)
```
This yields the output:
```
UserWarning: Parallelize plan key 'xd_emb' could not be resolved: no submodule matching token 'xd_emb' in module ImageEmbeddings(
  (tok_emb): Embedding(256, 128)
  (row_emb): Embedding(32, 128)
  (col_emb): Embedding(32, 128)
  (chan_emb): Embedding(3, 128)
), skipping this plan entry.
```
As expected.    

We also pass in loss_parallel via job.config.parallelism.disable_loss_parallel.
Async TP is also supported via `maybe_enable_async_tp`. This blogpost dives into the details of how this is implemented in PyTorch: https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487/7  

## Model Compile
Enabled by setting `job_config.compile.enable` and adding "model" to `job_config.compile.components`. 
Compilation is performed blockwise, by looping over the model layers in `apply_compile`. 

## Activation Checkpointing
Selective Activation Checkpointing is performed via `apply_ac`, we pass in a predetermined list of operations we wish to checkpoint. 
We use matrix-matrix multiplication, spda (efficient, flash, and flex), reduce_scatter_tensor, and max.default for mixed precision.
Large dense matrix multiplications dominate activation memory, recomputing is cheap since GEMM kernels are highly optimized. The same reasoning applies for SPDA and other attention variants. Collective communication ops like reduce scatter can be overlapped and recomputed cheaply, another way of directly reducing memory overhead. 

## Data Parallelism & Optimizer Sharding (FSDP) 
If `parallel_dims.fsdp_enabled` is set to `False`, only data parallel is applied via `apply_ddp`. This function checks if compilation is enabled and sets respective optimizations, then replicates the model across ranks with a fixed bucket cap of 100mb via `torch.distributed._composable.replicate`.  
If `fsdp_enabled` we apply data parallelism with FSDP2 via `apply_fsdp`. We must specify the model, DeviceMesh, parameter data types, and reduction operation data types, as well as provide info about pipeline parallelism (enabled/disabled), CPU offloading, and the policy that should be used for resharding after the forward pass (default, never, always). 

### HSDP
TODO: Read up on this

### Context Parallelism
Context Parallelism is only used indirectly in parallelization of the model, since we add it to the model via an optional context manager in train.py

### CPU Offloading
Set in `apply_fsdp` if cpu_offload is set to `True`. Defaults to `torch.distributed.fsdp.CPUOffloadPolicy` which is just a thin helper class that sets `pin_memory` to `True` such that communication can be more efficiently overlapped with computation. Should be set to `False` if CPU memory is insufficient. See https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_api.py for details.  

# Notes on train.py

## Float8
https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323  
https://arxiv.org/pdf/2209.05433  
https://github.com/pytorch/ao/tree/main/torchao/float8  
https://github.com/pytorch/torchtitan/blob/main/docs/float8.md  

## Context Parallelism
https://github.com/zhuzilin/ring-flash-attention  
stripe_flash_attn_func to use https://arxiv.org/abs/2311.09431  
TorchTitan implements Pass-KV Ring Attention https://arxiv.org/pdf/2411.01783  
https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082
https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/experimental/_attention.py  
_cp_ptions.enable_load_balance
