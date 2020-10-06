import numpy as np
from sklearn import datasets
import torch

from iris import Net, load_dataset


if __name__ == '__main__':

    # データセットを取得
    _, _, test_x, test_y = load_dataset()

    # モデルのインスタンスを生成
    model = Net()

    # 学習結果をロード
    model.load_state_dict(torch.load('my_iris_model'))
    model.eval()

    # 検証
    with torch.no_grad():
        output = model(test_x)
        ans = torch.argmax(output, 1)
        print(((test_y == ans).sum().float() / len(ans)).item())
