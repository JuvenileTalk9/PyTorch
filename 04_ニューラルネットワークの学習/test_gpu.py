import numpy as np
from sklearn import datasets
import torch

from iris import Net, load_dataset


if __name__ == '__main__':

    # データセットを取得
    _, _, test_x, test_y = load_dataset()

    # GPU利用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_x, test_y = test_x.to(device), test_y.to(device)

    # モデルのインスタンスを生成
    model = Net().to(device)

    # 学習結果をロード
    model.load_state_dict(torch.load('my_iris_model'))
    model.eval()

    # 検証
    with torch.no_grad():
        output = model(test_x)
        ans = torch.argmax(output, 1)
        print(((test_y == ans).sum().float() / len(ans)).item())
