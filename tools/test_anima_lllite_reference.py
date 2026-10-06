"""Generate independent fixtures from the reviewed upstream LLLite classes."""
import ast
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
model, reference, output = map(Path, sys.argv[1:])
output.mkdir(parents=True, exist_ok=True)
tree = ast.parse(reference.read_text())
selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            and node.name in {"_gn", "_ResBlock", "_ASPP", "_Conditioning1", "LLLiteModuleDiT"}]
assert len(selected) == 5
namespace = dict(torch=torch, nn=nn, F=F, Optional=Optional, Tuple=Tuple, ASPP_DEFAULT_DILATIONS=(1,2,4,8))
exec(compile(ast.Module(body=selected, type_ignores=[]), str(reference), "exec"), namespace)
weights = {key: value.float() for key, value in load_file(model).items()}
conditioning = {key.removeprefix("lllite_conditioning1."): value for key, value in weights.items() if key.startswith("lllite_conditioning1.")}
dim, embedding = conditioning["conv3.weight"].shape[0], conditioning["proj.weight"].shape[0]
residuals = sum(key.endswith("conv1.weight") and key.startswith("resblocks.") for key in conditioning)
encoder = namespace["_Conditioning1"](dim, embedding, residuals).eval()
encoder.load_state_dict(conditioning, strict=True)

class IdentityProjection(nn.Module):
    in_features = 2048
    def forward(self, x): return x

prefix = "lllite_dit_blocks_0_self_attn_q_proj."
params = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
depth = params.pop("depth_embed")
module = namespace["LLLiteModuleDiT"]("parity", IdentityProjection(), embedding, params["down.weight"].shape[0]).eval()
module.load_state_dict(params, strict=True)
module.org_forward = lambda x: x
torch.manual_seed(42)
image = torch.rand(1,3,64,64) * 2 - 1
x = torch.randn(1,16,2048)
with torch.no_grad():
    condition = encoder(image)
    module.cond_emb, module.depth_emb = condition, depth
    result = module(x)
for name, tensor in [("image", image), ("x", x), ("reference-condition", condition), ("reference-output", result)]:
    tensor.numpy().astype(np.float32).tofile(output / (name + ".f32"))
print("Reference fixtures:", tuple(condition.shape), tuple(result.shape), "residual RMS", (result-x).square().mean().sqrt().item())
