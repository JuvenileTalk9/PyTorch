# MobileNet-SSD

ここでは、torchvisionを利用してMobileNet-SSDをファインチューニングし、推論を実行します。

## プログラム全体の流れ

はじめに、学習モデルができるまでに必要な流れを示します。

1. データセットの準備
    - データセットの読み込み
    - Data Augmentation
2. モデルの定義
3. 損失関数の定義
4. 学習コードの作成
5. （オプション）推論コードの作成

PyTorchのSSD実装は[ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)がオリジナルとなっていますが、VGG-SSDのみの対応となっているため、VGGの他にMobileNetやAlexNetなどにも対応している[pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)を参考にします。

## 1. データセットの準備

学習・テストに必要なデータセットを読み込み、学習がうまく収束するためにデータを水増ししたり、モデルに合わせて入力フォーマットを変換する方法を示します。

### 1.1 データセットの読み込み

原論文では[PASCAL VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)、[PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)、[COCO2014](https://cocodataset.org/#home)を使用しており、データセットごとにアノテーション情報などのフォーマットが異なるため、形式ごとに読み込むコードを変える必要があります。ここでは、データセットはすべてVOCと同じ形式になっていることを前提として、VOC2007とVOC2012の2つのデータセットを使います。

PyTorchではデータセットは```Dataset```クラスを継承したデータセットクラスとして定義するか、```torch.utils.data.ConcatDataset```でデータセットクラスを生成する必要があり、いずれの場合もデータセット内に含まれるデータ数を返す```__len__```関数と```index```に対応する画像と正解ラベルを返す```__getitem__```関数を含む必要があります。

今回はVOC形式のデータセットを読み込む専用クラス```VOCDataset```を作成しました。

```python
class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        classes = []
        if os.path.isfile(label_file):
            with open(label_file, 'r') as infile:
                for line in infile:
                    classes.append(line.rstrip())

            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes  = [ elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor')


        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
```

### 1.2 Data Augmentation

Data Augmentationはデータセットの画像に対してリサイズや画素変換などの画像変換を実施した画像を別のデータとすることで、データセットを水増しする手法です。Data Augmentaionについては原論文は必ずしも必要でないオプション処理としていますが、

> Data augmentation is crucial（Data augmenmtationはとても重要だ！）

と述べております。実際にはData Augmentationなしの場合は学習がうまく収束しないことが多いため、実質的に必須処理になると思います。

上記の```VOCDataset```クラスは```__init__```の引数```transform```でData Augmentationを実行するクラスを指定できるようにしています。ここではData Augmentationとして以下を実装します。

|名称|説明|
|:--|:--|
|Random Contrast|画素に対して0.5～1.5までの値をランダムに乗算する|
|Random Saturation|画像をRGBからHSVに変換し、彩度（Saturation）に対して0.5～1.5までの値をランダムに乗算する|
|Random Hue|画像をRGBからHSVに変換し、色相（Hoe）に対して-18～18までの値をランダムに乗算する|
|Random Expand|画像をランダムに拡大する|
|Random Sample Crop|画像からランダムな領域をサンプリングする|
|Random Mirror|50%の確率で画像を反転する|

また、モデルへ入力可能かフォーマットに画像を変換する前処理もここで実装します。ここでは前処理として以下を実装します。

|名称|説明|
|:--|:--|
|Resize|300x300サイズにリサイズする|
|Normalization|画素の取りうる値が0～255から0~1になるように正規化する|
|To Tensor|numpy.ndarray形式からtorch.Tensor形式に変換|

プログラム全体はこちらにあります。

[data_preprocessing.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/08_MobileNet-SSD/pytorch-ssd/models/data_preprocessing.py)

[transforms.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/08_MobileNet-SSD/pytorch-ssd/transforms/transforms.py)

## 2. モデルの定義

MobileNet-SSDの全体は以下の構造になっています。

![mobilenet-ssd.jpg](https://github.com/JuvenileTalk9/PyTorch/tree/master/08_MobileNet-SSD/mobilenet-ssd.jpg)

これらの層を1つ1つ実装し、連結させたコードが以下になります。

[mobilenetv1_ssd.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/08_MobileNet-SSD/pytorch-ssd/models/mobilenetv1_ssd.py)

## 3. 損失関数の定義

モデルのパラメータを求めて更新するためには、何かしらの損失関数が必要です。下記図のように、SSDではlossをBBoxの距離の誤差```lossL```とラベルの誤差```lossC```の和として誤差を定義しています。ここで、```lossC```は正解と推定されたラベルの差なので簡潔ですが、```lossL```の考え方は特殊です。図のように、正解のBBoxに対して1つ選択したPriorBoxの誤差```d0```と、推定されたBBoxに対して正解のときと同一のPriorBoxの誤差```d1```を計算し、```d0```と```d1```の誤差が```lossL```になっています。そのため、SSDは枠の位置を学習しているのではなく、枠からの距離を学習していることになります。

![loss](https://github.com/JuvenileTalk9/PyTorch/tree/master/08_MobileNet-SSD/loss.jpg)

今回は損失関数は```MultiboxLoss```クラスで定義しました。

```python
class MultiboxLoss(nn.Module):
    def __init__(self, priors, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors.to(device)

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss/num_pos, classification_loss/num_pos
```

## 4. 学習コードの作成

学習のためのコードを作ります。PyTorchは学習プログラムはほぼ固定のため、説明は省略します。プログラム全体は以下をご参照ください。

[train.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/08_MobileNet-SSD/pytorch-ssd/train.py)

## 5. （オプション）推論コードの作成

必要に応じて学習したモデルを実行するコードを作ります。

[eval.py](https://github.com/JuvenileTalk9/PyTorch/blob/master/08_MobileNet-SSD/pytorch-ssd/eval.py)

[目次へ戻る](https://github.com/JuvenileTalk9/PyTorch)
