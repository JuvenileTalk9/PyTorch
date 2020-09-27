import torch

def f(x):
    """最小解を求めたい関数"""
    return (x[0] - 2 * x[1] - 1)**2 +(x[1] * x[2] - 1)**2 + 1

def f_grad(x):
    """変数がある値のときの微分値を求める関数"""
    z = f(x)
    z.backward()
    return x.grad

if __name__ == '__main__':
    # 初期値を決める（1, 2, 3）
    x = torch.tensor([1., 2., 3.], requires_grad=True)
    for i in range(50):
        # 微分値を求める
        x = x - 0.1 * f_grad(x)
        # 微分値に応じて少し移動
        x = x.detach().requires_grad_(True)
        print('x={},  f={}'.format(x.data, f(x).item()))