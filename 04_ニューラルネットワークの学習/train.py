import torch
import torch.nn as nn
import torch.optim as optim

from iris import Net, load_dataset


if __name__ == '__main__':

    # データセットを取得
    train_x, train_y, _, _ = load_dataset()

    # モデルのインスタンスを生成
    model = Net()

    # 最適化アルゴリズムを定義
    optimizer = optim.Adam(model.parameters(), lr=0.1)

    # 損失関数を定義
    criterion = nn.CrossEntropyLoss()

    # 学習
    for epoch in range(2000):
        output = model(train_x)
        loss = criterion(output, train_y)
        print(epoch, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 学習結果を保存
    torch.save(model.state_dict(), 'my_iris_model')
