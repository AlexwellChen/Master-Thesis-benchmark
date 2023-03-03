import torch
from adan import Adan
import argparse

def get_fake_parameters(n_params=10, size=512):
    params = []
    for i in range(n_params):
        tensor = torch.randn(size, size, requires_grad=True)
        tensor.grad = torch.randn(size, size)
        params.append(tensor)
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreach', type=bool, default=False)
    parser.add_argument('--fused', type=bool, default=False)
    args = parser.parse_args()
    
    params = get_fake_parameters()
    optimizer = Adan(params=params, foreach=args.foreach, fused=args.fused)
    for i in range(10):
        optimizer.step()
