

from safetensors.torch import load

file_path = "ckpt_diffusion/stats.safetensors"
with open(file_path, "rb") as f:
    data = f.read()

loaded = load(data)