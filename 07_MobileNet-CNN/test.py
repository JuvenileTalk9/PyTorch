import torch
import torch.nn as nn
import torchvision


if __name__ == '__main__':

    # GPU・CPU選択
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # テストデータをダウンロード
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    # ラベル一覧取得
    classes = testset.classes
    num_classes = len(classes)
    print('num_classes: {}'.format(num_classes))
    print('classes: {}'.format(classes))

    # モデル生成
    model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # 読み込み
    model_path = './mobilenet.pth'
    model.load_state_dict(torch.load(model_path))

    # 推論
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)
    print('ground_truth: {}'.format([classes[labels[i]] for i in range(4)]))
    print('detected_label: {}'.format([classes[predicted[i]] for i in range(4)]))
