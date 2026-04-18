"""
COPT — Forward pass of a small neural network on the GPU.

Architecture (Figure 3):
    Input  : 4 features
    Hidden : 5 neurons, ReLU activation
    Output : 1 neuron, linear (no activation)
Batch size: 16
"""

import sys
import torch
import torch.nn as nn

# ── 1. Device detection ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version     : {torch.version.cuda}")
    print(f"PyTorch version  : {torch.__version__}")
else:
    print("No CUDA GPU found — this script requires a CUDA-capable GPU.")
    print(f"(PyTorch version: {torch.__version__}, device: {device})")
    sys.exit(0)

# ── 2. Network definition ─────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(4, 5),   # hidden layer: 4 inputs -> 5 neurons
    nn.ReLU(),         # ReLU activation
    nn.Linear(5, 1),   # output layer: 5 -> 1, no activation
)
model.to(device)
print(f"\nModel architecture:\n{model}")
print(f"Model device     : {next(model.parameters()).device}")

# ── 3. Forward pass ───────────────────────────────────────────────────────────
batch = torch.randn(16, 4, device=device)   # [16, 4] random input on GPU
print(f"\nInput  shape : {list(batch.shape)}")
print(f"Input  device: {batch.device}")

output = model(batch)                        # forward pass
print(f"\nOutput shape : {list(output.shape)}")
print(f"Output device: {output.device}")

assert output.shape == (16, 1), f"Unexpected output shape: {output.shape}"
print("\nShape check passed: output is [16, 1]  OK")
