from re import X
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64
lr = 0.001
num_epochs = 100
patience = 10
valid_split = 0.1

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

# Split into training and validation sets
num_train = int((1 - valid_split) * train_data.shape[0])
valid_data = train_data[num_train:] # shape: (45000, 3072)
valid_labels = train_labels[num_train:] # shape: (5000,)
train_data = train_data[:num_train] # shape: (45000, 3072)
train_labels = train_labels[:num_train] # shape: (45000,)

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
train_data = train_data.reshape(45000, 3, 32, 32) / 255.0
valid_data = valid_data.reshape(5000, 3, 32, 32) / 255.0
test_data = test_data.reshape(10000, 3, 32, 32) / 255.0
X_train = torch.tensor(train_data, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_valid = torch.tensor(valid_data, dtype=torch.float32)
y_valid = torch.tensor(valid_labels, dtype=torch.long)
X_test = torch.tensor(test_data, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False)
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
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 64, 16, 16)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1), # input: 64 output: 64
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1), # input: 64 output: 192
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 192, 8, 8)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32), # input: 192 output: 64 + 128 + 32 + 32 = 256
            Inception(256, 128, (128, 192), (32, 64), 64), # input: 256 output: 128 + 192 + 64 + 64 = 448
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 448, 4, 4)
        )
        self.b4 = nn.Sequential(
            Inception(448, 192, (96, 208), (16, 48), 64), # input: 448 output: 192 + 208 + 48 + 64 = 512
            Inception(512, 160, (112, 224), (24, 64), 64), # input: 512 output: 160 + 224 + 64 + 64 = 512
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (batch_size, 512, 2, 2)
        )
        self.b5 = nn.Sequential(
            Inception(512, 128, (128, 256), (24, 64), 64), # input: 512 output: 128 + 256 + 64 + 64 = 512
            Inception(512, 256, (160, 320), (32, 128), 128), # input: 512 output: 256 + 320 + 128 + 128 = 832
            nn.AdaptiveAvgPool2d((1, 1)),                    # output: (batch_size, 1024, 1, 1)(相当于1x1图片)
            nn.Flatten()
        )
        self.fc = nn.Linear(832, 10)
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

# early stopping variables
best_valid_loss = float('inf')
best_model_state = None
epochs_without_improvement = 0

# Training loop with early stopping
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    valid_loss = 0
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            valid_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            valid_correct += (preds == y_batch).sum().item()
            valid_total += y_batch.size(0)
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_acc = 100 * valid_correct / valid_total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
    
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        best_model_state = model.state_dict().copy()
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print(f'\nEarly stop.')
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Final evaluation on test set
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
print(f'\nFinal Accuracy: {100 * correct / total:.2f}%')