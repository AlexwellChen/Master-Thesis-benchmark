import torch
from adan import Adan
import argparse

def get_fake_parameters(n_params=10, size=512, fp16=False):
    params = []
    for i in range(n_params):
        if not fp16:
            tensor = torch.randn(size, size, requires_grad=True, device='cuda')   
            tensor.grad = torch.randn(size, size, device='cuda')
            params.append(tensor)
        else:
            tensor = torch.randn(size, size, requires_grad=True, device='cuda')   
            tensor.grad = torch.randn(size, size, device='cuda', dtype=torch.float16)
            params.append(tensor)
    return params

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foreach', type=str, default='False')
    parser.add_argument('--fused', type=str, default='False')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--n_params', type=int, default=10)
    parser.add_argument('--fp16', type=str, default='False')
    args = parser.parse_args()

    if args.foreach == 'True':
        args.foreach = True
    else:
        args.foreach = False

    if args.fused == 'True':
        args.fused = True
    else:
        args.fused = False
    
    if args.fp16 == 'True':
        args.fp16 = True
    else:
        args.fp16 = False

    params = get_fake_parameters(size=args.size, n_params=args.n_params, fp16=args.fp16)
    optimizer = Adan(params=params, foreach=args.foreach, fused=args.fused)
    
    torch.cuda.cudart().cudaProfilerStart()
    optimizer.step()
    torch.cuda.cudart().cudaProfilerStop()
