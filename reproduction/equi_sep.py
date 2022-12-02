import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import scipy
import scipy.linalg
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")


class FashionDataset(Dataset):
    def __init__(self, data, transform=None):
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        label, image = [], []

        for i in self.fashion_MNIST:
            label.append(i[0])
            image.append(i[1:])
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]

        if self.transform is not None:
            # transfrom the numpy array to PIL image before the transform function
            pil_image = Image.fromarray(np.uint8(image))
            image = self.transform(pil_image)

        return image, label


AlexTransform = transforms.Compose([
    transforms.Resize((10, 10)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


class FNN(nn.Module):
    def __init__(self, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.input_size = 100
        self.hidden_size = 100
        self.output_size = 10
        self.inputs = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(inplace=True)
        )
        layers = []
        for _ in range(self.layer_num):
            layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.ReLU(inplace=True)
                )
            )
        self.layers = nn.ModuleList(layers)
        self.outputs = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.inputs(out)
        for i in range(self.layer_num):
            out = self.layers[i](out)
        out = self.outputs(out)
        return out

    def compute_layer_output(self, x, l):
        out = x.view(x.size(0), -1)
        if l == 0:
            return out
        out = self.inputs(out)
        if l == 1:
            return out
        for i in range(l - 1):
            out = self.layers[i](out)
        return out


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 2 == 0:
            print("Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def compute_d(features, labels):
    features = copy.deepcopy(features)
    labels = copy.deepcopy(labels)
    dim = len(features[0])
    per_class_map = {}
    for i in range(len(labels)):
        if labels[i] not in per_class_map:
            per_class_map[labels[i]] = []
        per_class_map[labels[i]].append(features[i])

    global_mean = np.zeros(dim)
    for i in range(10):
        global_mean += np.mean(np.array(per_class_map[i]), axis=0) * len(per_class_map[i])
    global_mean /= len(features)

    ss_b = np.zeros((dim, dim))
    for i in range(10):
        normalized_class_features = np.mean(np.array(per_class_map[i]), axis=0) - global_mean
        ss_b += len(per_class_map[i]) * np.matmul(
            normalized_class_features.reshape(-1, 1),
            normalized_class_features.reshape(1, -1)
        )
    ss_b /= len(features)
    ss_b_inv = scipy.linalg.pinv(ss_b)

    ss_w = np.zeros((dim, dim))
    for i in range(10):
        class_average = np.mean(per_class_map[i], axis=0)
        for feature in per_class_map[i]:
            ss_w += np.matmul(
                (feature - class_average).reshape(-1, 1),
                (feature - class_average).reshape(1, -1)
            )
    ss_w /= len(features)
    d = np.trace(np.matmul(ss_w, ss_b_inv))
    return d


def draw(nums, output_path):
    nums = np.log10(np.array(nums))
    plt.scatter(np.array(list(range(len(nums)))), nums)
    plt.ylim([-4, 4])
    plt.savefig(output_path)
    plt.close()


def produce_rep(model, data_loader, layer_idx):
    all_features = []
    all_labels = []
    for item in data_loader:
        input = item[0].to(DEVICE)
        output = item[1].to(DEVICE)
        features = model.compute_layer_output(input, layer_idx)
        all_features.extend(features.detach().cpu().numpy().tolist())
        all_labels.extend(output.detach().cpu().numpy().tolist())
    d = compute_d(np.array(all_features), np.array(all_labels))
    return d


def experiment(learning_rate, layer_num, output_path):
    model = FNN(layer_num).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    train_csv = pd.read_csv('fashion-mnist_train_sampled.csv')
    train_loader = DataLoader(
        FashionDataset(train_csv, transform=AlexTransform), batch_size=128, shuffle=True,
    )
    epochs_num = 600

    for epoch in range(1, epochs_num+1):
        if epoch % 200 == 0:
            learning_rate /= 10.0
            for g in optimizer.param_groups:
                g['lr'] = learning_rate
        train(model, DEVICE, train_loader, optimizer, criterion, epoch)

    train_csv = pd.read_csv('fashion-mnist_train_sampled.csv')
    train_loader = DataLoader(
        FashionDataset(train_csv, transform=AlexTransform),
        batch_size=100, shuffle=True
    )

    outputs = []
    for i in range(0, layer_num + 2):
        outputs.append(produce_rep(model, train_loader, i))

    draw(outputs, output_path)


experiment(3e-4, 18, "/scratch1/xzhou45/figs/fig_20_3e-4.png")
experiment(1e-3, 6, "/scratch1/xzhou45/figs/fig_8_1e-3.png")
experiment(1e-3, 2, "/scratch1/xzhou45/figs/fig_4_1e-3.png")
