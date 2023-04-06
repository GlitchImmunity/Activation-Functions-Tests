import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu
        out = torch.cat((out(x), out(-x)), dim=1)
        return out

class PGLU(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=1, padding=0, pool=False):
        super(PGLU, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, gate, output):
        gate = self.conv(gate)
        if self.pool is not None:
            gate = self.pool(gate)
        gate = self.relu(gate)
        gate = gate > 0
        return output * gate

class OGLU(nn.Module):
    def __init__(self, channels, kernel_size=1, padding=0):
        super(OGLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, output):
        gate = self.conv(output)
        gate = self.relu(gate)
        gate = gate > 0
        return output * gate

class APGLU(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=1, padding=0, pool=False):
        super(APGLU, self).__init__()
        self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, gate, output):
        gate = self.conv(gate)
        if self.pool is not None:
            gate = self.pool(gate)
        gate = self.sigmoid(gate)
        return output * gate

class AOGLU(nn.Module):
    def __init__(self, channels, kernel_size=1, padding=0):
        super(AOGLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, output):
        gate = self.conv(output)
        gate = self.sigmoid(gate)
        return output * gate

class Net(pl.LightningModule):
    def __init__(self, lr=3e-4,
                 act=[nn.ReLU(), nn.ReLU(), nn.ReLU(), nn.ReLU()],
                 use_optim=False):
        super(Net, self).__init__()
        self.lr = lr
        self.use_optim = use_optim
        self.max_acc = 0

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.act1 = act[0]
        self.d1 = nn.Dropout(0.2)

        if isinstance(act[0], CReLU):
            self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        else:
            self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.act2 = act[1]
        self.d2 = nn.Dropout(0.2)

        if isinstance(act[1], CReLU):
            self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        else:
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.act3 = act[2]
        self.d3 = nn.Dropout(0.2)

        if isinstance(act[2], CReLU):
            self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        else:
            self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bn4 = nn.BatchNorm2d(256)
        self.act4 = act[3]
        self.d4 = nn.Dropout(0.2)

        if isinstance(act[3], CReLU):
            self.fc1 = nn.Linear(512 * 2 * 2, 10)
        else:
            self.fc1 = nn.Linear(256 * 2 * 2, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.bn1(out)
        if isinstance(self.act1, PGLU) or isinstance(self.act1, APGLU):
            out = self.act1(x, out)
        else:
            out = self.act1(out)
        out2 = self.d1(out)

        out = self.conv2(out2)
        out = self.pool2(out)
        out = self.bn2(out)
        if isinstance(self.act2, PGLU) or isinstance(self.act2, APGLU):
            out = self.act2(out2, out)
        else:
            out = self.act2(out)
        out3 = self.d2(out)

        out = self.conv3(out3)
        out = self.pool3(out)
        out = self.bn3(out)
        if isinstance(self.act3, PGLU) or isinstance(self.act3, APGLU):
            out = self.act3(out3, out)
        else:
            out = self.act3(out)
        out4 = self.d3(out)

        out = self.conv4(out4)
        out = self.pool4(out)
        out = self.bn4(out)
        if isinstance(self.act4, PGLU) or isinstance(self.act4, APGLU):
            out = self.act4(out4, out)
        else:
            out = self.act4(out)
        out = self.d4(out)

        if isinstance(self.act4, CReLU):
            out = out.view(-1, 512*2*2)
        else:
            out = out.view(-1, 256*2*2)
        out = self.fc1(out)
        return out

    def configure_optimizers(self):
        if self.use_optim:
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
            exp_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
            return [optimizer], [exp_lr_scheduler]
        else:
            return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor()), batch_size=5000, shuffle=True)

    def val_dataloader(self):
        return DataLoader(CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=5000, shuffle=False)

    def test_dataloader(self):
        return DataLoader(CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor()), batch_size=5000, shuffle=False)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, prog_bar=False)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_acc', accuracy, prog_bar=False)
        if accuracy > self.max_acc:
            self.max_acc = accuracy
            self.log('max_acc', accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss