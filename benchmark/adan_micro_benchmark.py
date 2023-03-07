import torch
from adan import Adan
import argparse

def get_fake_parameters(n_params=10, size=512):
    params = []
    for i in range(n_params):
        tensor = torch.randn(size, size, requires_grad=True, device='cuda')
        tensor.grad = torch.randn(size, size, device='cuda')
        params.append(tensor)
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreach', type=str, default='False')
    parser.add_argument('--fused', type=str, default='False')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_params', type=int, default=10)
    args = parser.parse_args()

    if args.foreach == 'True':
        args.foreach = True
    else:
        args.foreach = False

    if args.fused == 'True':
        args.fused = True
    else:
        args.fused = False

    params = get_fake_parameters(size=args.size, n_params=args.n_params)
    optimizer = Adan(params=params, foreach=args.foreach, fused=args.fused)
    torch.cuda.cudart().cudaProfilerStart()
    for i in range(10):
        optimizer.step()
    torch.cuda.cudart().cudaProfilerStop()
