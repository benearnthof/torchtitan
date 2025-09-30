python3 -m venv tt && source tt/bin/activate
python -m pip install --upgrade pip

# we need the nightly builds of pytorch for experimental features to work
# pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
# pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r ./torchtitan/.ci/docker/requirements.txt

pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu128
