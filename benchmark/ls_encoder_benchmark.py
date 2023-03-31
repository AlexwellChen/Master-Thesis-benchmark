import torch
import torch.nn as nn
from ls_module.ls_hf_transformer_layer import LSHFTransformerEncoderLayer
# nvtx
import torch.cuda.nvtx as nvtx

# Define the transformer encoder
pt_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=1)
ls_encoder = nn.TransformerEncoder(LSHFTransformerEncoderLayer(d_model=512, nhead=8), num_layers=1)

# Generate fake data
batch_size = 32
seq_len = 100
input_dim = 512
fake_data = torch.randn(batch_size, seq_len, input_dim)

# Perform forward and backward propagation
for i in range(10):
    # nvtx forward
    nvtx.range_push("pt_forward")
    output = pt_encoder(fake_data)
    nvtx.range_pop()
    loss = output.mean()
    # nvtx backward
    nvtx.range_push("pt_backward")
    loss.backward()
    nvtx.range_pop()

for i in range(10):
    # nvtx forward
    nvtx.range_push("ls_forward")
    output = ls_encoder(fake_data)
    nvtx.range_pop()
    loss = output.mean()
    # nvtx backward
    nvtx.range_push("ls_backward")
    loss.backward()
    nvtx.range_pop()