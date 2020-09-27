# Tensorによる基本的な演算

本章では、PyTorchの基本的なデータ型であるTensor型について学びます。

## Tensorの作成

```tourch.tensor```で配列のTensorオブジェクトが作れます。Numpyのarrayと似ています。

```python
>>> torch.tensor([1, 2, 3])
tensor([1, 2, 3])
>>> np.array([1, 2, 3])
array([1, 2, 3])

>>> torch.tensor([[0, 1, 2], [3, 4, 5]])
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> np.array([[0, 1, 2], [3, 4, 5]])
array([[0, 1, 2],
       [3, 4, 5]])
```

```torch.Tensor```を使用してもオブジェクトが作れますが、```torch.tensor```がlong型のTensorオブジェクトを作成することに対して、```torch.Tensor```はfloat型になります。

```python
>>> torch.Tensor([1, 2, 3])
tensor([1., 2., 3.])
```

```torch.Tensor```は内容を初期化せずに形状だけを指定することも可能です。下の例では、2×3の行列を定義しています。この場合、内容は0になるとは限りません。

```python
>>> torch.Tensor(2, 3)
tensor([[0., 0., 0.],
        [0., 0., 0.]])
>>> torch.Tensor(3)
tensor([0., 0., 0.])
```

形状を変える場合は、```reshape```を使います。

```python
>>> torch.arange(6).reshape(2, 3)
tensor([[0, 1, 2],
        [3, 4, 5]])
>>> np.arange(6).reshape(2, 3)
array([[0, 1, 2],
       [3, 4, 5]])
```

## Tensorの四則演算

Tensorと整数の四則演算の結果は、同じサイズのTensorになります。

```python
>>> a = torch.tensor([1, 2, 3])
>>> a + 2
tensor([3, 4, 5])
>>> a - 3
tensor([-2, -1,  0])
>>> a * 5
tensor([ 5, 10, 15])
>>> a // 2
tensor([0, 1, 1])
```

TensorとTensorの四則演算の結果は、同じサイズのTensorになります。必ず同じサイズのTensor同士で演算する必要があります。

```python
>>> a = torch.arange(6).reshape(2, 3)
>>> b = a + 2
>>> a + b
tensor([[ 2,  4,  6],
        [ 8, 10, 12]])
>>> a - b
tensor([[-2, -2, -2],
        [-2, -2, -2]])
>>> a * b
tensor([[ 0,  3,  8],
        [15, 24, 35]])
>>> a // b
tensor([[0, 0, 0],
        [0, 0, 0]])
```

## Tensorの行列積

ベクトル同士の内積や、行列とベクトルの乗算、行列同士の乗算は関数を使って計算します。

ベクトル同士の内積は```dot```や```matmul```を使います。

```python
>>> a = torch.tensor([1, 2, 3, 4])  # ベクトル  
>>> b = torch.tensor([5, 6, 7, 8])  # ベクトル
>>> torch.dot(a, b)
tensor(70)
>>> torch.matmul(a, b)
tensor(70)
```

行列とベクトルの乗算は```mv```や```matmul```を使います。

```python
>>> a = torch.arange(8).reshape(2, 4)  # 行列
>>> b = torch.tensor([5, 6, 7, 8])     # ベクトル
>>> torch.mv(a, b)
tensor([ 44, 148])
>>> torch.matmul(a, b)
tensor([ 44, 148])
```

行列同士の乗算は```mm```や```matmult```を使います。

```python
>>> a = torch.arange(8).reshape(2, 4)  # 行列
>>> b = torch.arange(8).reshape(4, 2)  # 行列
>>> torch.mm(a, b)
tensor([[28, 34],
        [76, 98]])
>>> torch.matmul(a, b)
tensor([[28, 34],
        [76, 98]])
```

## TensorとNumpyの相互変換

Tensorのをnumpyのarrayに変換するときは```numpy()```を使います。逆に、numpyのarrayをTensorに変換するときは```from_numpy()```を使います。

```python
>>> a = torch.tensor([1, 2, 3])
>>> type(a)
<class 'torch.Tensor'>
>>> a
tensor([1, 2, 3])

>>> b = a.numpy()
>>> type(b)
<class 'numpy.ndarray'>
>>> b
array([1, 2, 3], dtype=int64)

>>> c = torch.from_numpy(b)
>>> type(c)
<class 'torch.Tensor'>
>>> c
tensor([1, 2, 3])
```

## Tensorによる微分計算

PyTorchのTensorとNumpyのarrayは、これまでに見てきたように使い方について大きな違いはありません。最大の違いは、PyTorchのTensorは微分値を求める機能がありますが、Numpyのarrayは微分値を求める機能がありません。ディープラーニングにおいては変数の微分を求める処理が必須であり、これがNumpyではなくPyTorchを使う理由になります。

微分値を求めたい場合、tensorの定義時に```requires_grad```という属性値を```True```に設定します。```requires_grad```のデフォルト値は```False```なので、手動で設定する必要があります。

以下は、関数f：R<sup>3</sup>→Rの最小解を求める最急降下法のプログラムです。関数はプログラム中の```f```関数で定義しています。プログラム中で、```x = x.detach().requires_grad_(True)```とありますが、これは一度微分値を求めたら次に微分値を求める前に、一度```detach()```で微分の情報を外してから、再度```requires_grad_(True)```で微分値を求めるように設定する必要があるからです。

```python
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
```

このプログラムの実行結果は以下になりました。

```python
x=tensor([ 1.8000, -2.6000,  1.0000]),  f=49.959999084472656
x=tensor([ 0.6000,  0.5200, -0.8720]),  f=5.186089515686035
x=tensor([ 0.8880, -0.3095, -0.7208]),  f=1.8606034517288208
・・・（中略）・・・
x=tensor([-0.2068, -0.6060, -1.6389]),  f=1.0000731945037842
x=tensor([-0.2078, -0.6062, -1.6398]),  f=1.000056505203247
x=tensor([-0.2087, -0.6063, -1.6405]),  f=1.0000436305999756
```
だいたい```x=(-0.2, -0.6, -1.6)```で最小値```1```になることが求められました。

[目次へ戻る](https://github.com/JuvenileTalk9/PyTorch)
