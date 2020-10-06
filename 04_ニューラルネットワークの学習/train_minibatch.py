import numpy as np
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

    # バッチサイズの設定
    data_size = len(train_x)
    batch_size = 25

    # 学習
    for epoch in range(2000):
        idx = np.random.permutation(data_size)
        for batch in range(0, data_size, batch_size):
            train_x_batch = train_x[idx[batch:batch+batch_size]]
            train_y_batch = train_y[idx[batch:batch+batch_size]]
            output = model(train_x_batch)
            loss = criterion(output, train_y_batch)      
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 学習結果を保存
    torch.save(model.state_dict(), 'my_iris_model')
