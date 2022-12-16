import warnings
import os
import argparse

import numpy as np
import torch
import torch.nn as nn

from model.model import VGG16
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=str, default=0.01, help='Learning rate')
    parser.add_argument('--train_fn', type=str, default='./cifar10/', help='train set path')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=60, help='Epochs')
    parser.add_argument('--num_class', type=int, default=10, help="number class.")
    parser.add_argument('--valid_ration', type=float, default=0.1, help='valid set ration')
    args = parser.parse_args()

    data_transform = transforms.Compose(
        transforms=[transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616))])

    dataset = datasets.CIFAR10(root=args.train_fn, train=True, transform=data_transform, download=True)
    train_size = int((1 - args.valid_ration) * len(dataset))
    valid_size = len(dataset) - train_size
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True)

    network = VGG16(num_class=args.num_class, dropout=0.5).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=network.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    train_step = len(train_loader)
    valid_num = len(valid_set)
    best_acc = 0.0

    for epoch in range(args.epochs):
        network.train()
        step = 1
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            y_hat = network(images)
            loss = loss_fn(y_hat, labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch: [{epoch + 1} / {args.epochs}], Step: [{step} / {train_step}], Loss: [{loss}]')
            step += 1

        network.eval()
        with torch.no_grad():
            accuracy = 0.0
            loss = 0.0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                y_hat = network(images)
                loss += loss_fn(y_hat, labels)
                prediction = torch.max(y_hat, dim=1)[1]
                accuracy += (prediction == labels).sum().item()
            accuracy /= valid_num
            print(f'Epoch: [{epoch + 1} / {args.epochs}], loss: [{loss / valid_num:.4f}], accuracy: [{accuracy:.4f}]')
            if accuracy > best_acc:
                best_acc = accuracy
                print('Saving model to ./output/')
                # torch.save(network.state_dict(), f'./output/weights_{epoch}_{accuracy:.4f}.pth')
