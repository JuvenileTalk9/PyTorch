# DataLoaderの使い方

本章の目的は、DataLoaderを使ったミニバッチ学習のためのデータ生成について学びます。DataLoaderを使うことで、データの前処理、バッチ処理、シャッフルが自動的に行なえるようになり、プログラムが簡潔になることが期待できます。

## DataLoaderの実装手順

DataLoaderの実装は次の3ステップが必要です。

1. 前処理を行うクラスの作成
2. データセットを管理するクラスの作成
3. DataLoaderの作成

### 前処理を行うクラスの作成

まず、前処理を行うクラスを作成します。クラス名は任意の名前で構いませんが、必ず```__call__```メソッドをオーバーライドして、前処理を定義する必要があります。```__call__```の入力はデータ```x```とラベル```y```のタプルで、出力は前処理を通したデータ```x'```とラベル```y```のタプルです。

以下は、入力画像サイズを16x16に変換する例です。

```python
class PreProcess(object):
    '''画像を16x16サイズに変換する前処理'''
    def __init__(self):
        pass
    def __call__(self, xy):
        x, y = xy
        x = cv2.resize(x, (16, 16))
        return (x, y)
```

### データセットを管理するクラスの作成

次に、データセットを管理するクラスを作成します。クラス名は任意の名前で構いませんが、必ず```torch.uitls.data.Dataset```を検証する必要があります。また、データセットのデータ数を返す```__len__```メソッドと、データセットからデータを1つ取得する```__getitem__```メソッドを実装する必要があります。```__getitem__```の入力はデータセットかららデータを指定するインデックス番号です。

以下は、CIFAR10データセットを扱うデータセットクラスです。プログラムと同じディレクトリにデータセットファイル（data_batch_1～data_batch_5）を配置する必要があります。

```python
class MyDataset(Dataset):
    '''CIFAR10データセットを管理するDatasetクラス'''
    def __init__(self, prepro):
        data, labels = [], []
        # CIFAR10のファイル（事前にこのプログラムと同じディレクトリに配置する）
        cifar10_files = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                'data_batch_4', 'data_batch_5']
        for file in cifar10_files:
            dic = self._unpickle(file)
            # 複数の画像がまとめてバイナリで保存されているので、
            # 画像枚数xチャンネル数x縦幅x横幅の4次元に変換する。
            # そのあとでRGBからBGRに変換する。
            data.append(dic[b'data'].reshape(10000, 3, 32, 32)
                    .transpose(0, 2, 3, 1).astype('uint8'))
            labels.append(dic[b'labels'])

        self.train_x = np.concatenate(data)
        self.train_y = np.concatenate(labels)
        self.len = len(self.train_x)
        self.prepro = prepro

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.prepro((self.train_x[idx, :, :, :], self.train_y[idx]))

    def _unpickle(self, file):
        with open(file, 'rb') as f:
            dic = pickle.load(f, encoding='bytes')
        return dic
```

### DataLoaderの作成

DataLoaderは```torch.utils.data.DataLoader```クラスを使用します。```DataLoader```の第一引数は、先程作成したデータセットクラスのインスタンスです。その他、```batch_size```でバッチサイズを指定します。```shuffle```に```True```を指定することで、データセットが自動的にシャッフルされます。

dataloaderをfor文のイテラブルオブジェクトとしてしていすることで、データとラベルのタプルを取得できます。内容を確認すると、DataLoaderのインスタンス化時に指定したバッチサイズと同じサイズになっていることがわかります。

```python
prepro = PreProcess()
dataset = MyDataset(prepro)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for x, y in dataloader:
    print(x, y)
```

プログラム全体は以下を参照してください。

[dataloader.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/05_DataLoader%E3%81%AE%E4%BD%BF%E3%81%84%E6%96%B9/dataloader.py)

[目次へ戻る](https://github.com/JuvenileTalk9/PyTorch)
