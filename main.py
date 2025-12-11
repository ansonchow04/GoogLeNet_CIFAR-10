import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64
lr = 0.01
num_epochs = 10

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

train_data_file = 'data/cifar-10-batches-py/data_batch_'
test_data_file = 'data/cifar-10-batches-py/test_batch'

# unpickle the CIFAR-10 data files and load data
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Load training data
train_data = []
train_labels = []
for i in range(1, 6):
    dataFileName_ = train_data_file + str(i)
    batch = unpickle(dataFileName_)
    train_data.append(batch[b'data']) # len(train_data) = 5, train_data[0].shape = (10000, 3072), 32*32*3=3072, 10000 images per batch
    train_labels.append(batch[b'labels']) # train_labels[0] shape: (10000,), labels are integers from 0 to 9, representing the class of each image
train_data = np.vstack(train_data) # shape: (50000, 3072)
train_labels = np.hstack(train_labels) # shape: (50000,)

# Load test data
test_data = []
test_labels = []
dataFileName_ = test_data_file
batch = unpickle(dataFileName_)
test_data.append(batch[b'data']) # shape: (10000, 3072)
test_labels.append(batch[b'labels']) # shape: (10000,)
test_data = np.vstack(test_data) # shape: (10000, 3072)
test_labels = np.hstack(test_labels) # shape: (10000,)

# Preprocess data: reshape and normalize
train_data = train_data.reshape(50000, 3, 32, 32) / 255.0
test_data = test_data.reshape(10000, 3, 32, 32) / 255.0
X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# # Visualize an example image from the training data
# img_test = train_data.reshape(50000, 3, 32, 32)[8].transpose(1, 2, 0)
# plt.imshow(img_test)
# plt.axis('off')
# plt.show()

class Inception(nn.Module):
    def __init__(self, in_channels, oc1, oc2, oc3, oc4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 1x1 conv branch
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=oc1, kernel_size=1),
            nn.ReLU()
        )
        # 1x1 -> 3x3
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=oc2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=oc2[0], out_channels=oc2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 1x1 -> 5x5
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=oc3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=oc3[0], out_channels=oc3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        # 3x3 pool -> 1x1
        self.p4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=oc4, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        p1 = self.p1(x) # output shape: (batch_size, oc1, H, W)
        p2 = self.p2(x) # output shape: (batch_size, oc2[1], H, W)
        p3 = self.p3(x) # output shape: (batch_size, oc3[1], H, W)
        p4 = self.p4(x) # output shape: (batch_size, oc4, H, W)
        return torch.cat((p1, p2, p3, p4), dim=1) # output shape: (batch_size, oc1 + oc2[1] + oc3[1] + oc4, H, W)

class CIFAR10_GoogLeNet(nn.Module):
    def __init__(self):
        super(CIFAR10_GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1), # input: 3 output: 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 64, 64, 64)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), # input: 64 output: 64
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1), # input: 64 output: 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 192, 32, 32)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32), # input: 192 output: 64 + 128 + 32 + 32 = 256
            Inception(256, 128, (128, 192), (32, 96), 64), # input: 256 output: 128 + 192 + 96 + 64 = 480
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # output: (batch_size, 480, 32, 32)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64), # input: 480 output: 192 + 208 + 48 + 64 = 512
            Inception(512, 160, (112, 224), (24, 64), 64), # input: 512 output: 160 + 224 + 64 + 64 = 512
            Inception(512, 128, (128, 256), (24, 64), 64), # input: 512 output: 128 + 256 + 64 + 64 = 512
            Inception(512, 112, (144, 288), (32, 64), 64), # input: 512 output: 112 + 288 + 64 + 64 = 528
            Inception(528, 256, (160, 320), (32, 128), 128), # input: 528 output: 256 + 320 + 128 + 128 = 832
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # output: (batch_size, 832, 32, 32)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128), # input: 832 output: 256 + 320 + 128 + 128 = 832
            Inception(832, 384, (192, 384), (48, 128), 128), # input: 832 output: 384 + 384 + 128 + 128 = 1024
            nn.AdaptiveAvgPool2d((1, 1)),                    # output: (batch_size, 1024, 1, 1)(相当于1x1图片)
            nn.Flatten()
        )
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.fc(x)
        return x

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10_GoogLeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for X_batch, y_batch in tqdm(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
print(f'Test Accuracy: {100 * correct / total:.2f}%')