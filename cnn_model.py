import torch.nn as nn


class MnistCNNModel(nn.Module):
    def __init__(self):
        super(MnistCNNModel, self).__init__()
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=4)
        # x = F.relu(self.mp(self.conv1(x)))
        self.fc1 = nn.Linear(87616, 8)
        self.fc2 = nn.Linear(8, 1000)
        self.fc3 = nn.Linear(1000, 8)
        # self.sm = nn.Softmax()

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = out.view(out.size(0), -1)
        out = self.relu1(out)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
