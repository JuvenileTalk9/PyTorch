import torch

def f(x):
    return 2 * x * x + 5

def f_grad(x):
    g1 = 4 * x
    return torch.Tensor([g1])

if __name__ == '__main__':
    x = torch.Tensor([100])
    for i in range(50):
        x = x - 0.1 * f_grad(x)
        print('x = {},  f = {}'.format(x, f(x)))
