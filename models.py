from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch import flatten


class CNNBasic(Module):
    def __init__(self, numChannels=1, classes=1):
        # call the parent constructor
        super(CNNBasic, self).__init__()

        self.conv1 = Conv2d(in_channels=numChannels, out_channels=12,
            kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = Conv2d(in_channels=12, out_channels=24,
            kernel_size=(5, 5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = Conv2d(in_channels=24, out_channels=48,
            kernel_size=(5, 5))
        self.relu3 = ReLU()
        self.maxpool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4 = Conv2d(in_channels=48, out_channels=96,
            kernel_size=(5, 5))
        self.relu4 = ReLU()
        self.maxpool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc1 = Linear(in_features=75264, out_features=1000)
        self.relu5 = ReLU()

        self.fc2 = Linear(in_features=1000, out_features=classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu5(x)
        output = self.fc2(x)
        # return the output predictions
        return output