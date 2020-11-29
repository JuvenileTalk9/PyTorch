import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


if __name__ == '__main__':

    # GPU・CPU選択
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # 学習データをダウンロード
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # ラベル一覧取得
    classes = trainset.classes
    num_classes = len(classes)
    print('num_classes: {}'.format(num_classes))
    print('classes: {}'.format(classes))

    # モデル生成
    model = torchvision.models.mobilenet_v2(pretrained=True, progress=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model = model.to(device)

    # 学習パラメータ定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    num_epoch = 1000
    log_interval = 1000
    out_model_path = './mobilenet.pth'

    # 学習
    for epoch in range(num_epoch):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 1):
            inputs, labels = data
            outputs = model(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if not i % log_interval:
                print('epoch: {}/{}, batch: {}/{}, loss: {:.3f}'.format(epoch + 1, num_epoch, i, len(trainloader), running_loss / log_interval))
                running_loss = 0.0
        torch.save(model.state_dict(), '{}.{}'.format(out_model_path, epoch + 1))

    # 学習済みモデル出力
    torch.save(model.state_dict(), out_model_path)
    print('Finished Training: {}'.format(out_model_path))
