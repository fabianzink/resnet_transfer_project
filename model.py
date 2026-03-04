import torch
import torch.nn as nn

class cifar_cnn(nn.Module):
    def __init__(self):
        super().__init__()

        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxp3 = nn.MaxPool2d(kernel_size=2, stride=2)



        # fully connected layers
        self.linearlayer1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout1 = nn.Dropout(p=0.5)

        self.linearlayer2 = nn.Linear(in_features=1024, out_features=306)
        self.dropout2 = nn.Dropout(p=0.5)

        self.linearlayer3 = nn.Linear(in_features=306, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.maxp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.maxp3(x)
        x = x.view(-1, 2048)

        x = self.linearlayer1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.linearlayer2(x)
        x = torch.relu(x)
        x = self.dropout2(x)


        x = self.linearlayer3(x)

        return x