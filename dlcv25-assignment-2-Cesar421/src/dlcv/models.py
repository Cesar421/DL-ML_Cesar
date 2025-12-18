import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomizableNetwork(nn.Module):

    def __init__(self, conv_layers: int, filters_conv1: int, filters_conv2: int, filters_conv3: int,dense_units: int):
        super(CustomizableNetwork, self).__init__()

        # Ensure the number of convolutional layers is at least 2 and at most 3
        assert 2 <= conv_layers <= 3, "conv_layers must be 2 or 3"

        # Dictionary to hold all layers
        self.layers = nn.ModuleDict()

        # First convolutional block: Conv -> BatchNorm -> MaxPool
        # Input: 3 channels (RGB), Output: filters_conv1 channels
        self.layers['conv1'] = nn.Conv2d(in_channels=3, out_channels=filters_conv1, 
                                         kernel_size=3, padding=1)
        self.layers['bn1'] = nn.BatchNorm2d(num_features=filters_conv1)
        self.layers['pool1'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block: Conv -> BatchNorm -> MaxPool
        # Input: filters_conv1 channels, Output: filters_conv2 channels
        self.layers['conv2'] = nn.Conv2d(in_channels=filters_conv1, out_channels=filters_conv2,
                                         kernel_size=3, padding=1)
        self.layers['bn2'] = nn.BatchNorm2d(num_features=filters_conv2)
        self.layers['pool2'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional block (optional): Conv -> BatchNorm -> MaxPool
        # Only created if conv_layers == 3
        if conv_layers == 3:
            self.layers['conv3'] = nn.Conv2d(in_channels=filters_conv2, out_channels=filters_conv3,
                                             kernel_size=3, padding=1)
            self.layers['bn3'] = nn.BatchNorm2d(num_features=filters_conv3)
            self.layers['pool3'] = nn.MaxPool2d(kernel_size=2, stride=2)

        # Adaptive pooling layer
        # Converts any spatial size to 1x1, making the network flexible to different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Determine the number of input features for the first fully connected layer
        # After adaptive pooling, we have: batch_size x num_filters x 1 x 1
        # So the flattened size is: num_filters (from the last conv layer)
        if conv_layers == 3:
            fc_input_features = filters_conv3
        else:
            fc_input_features = filters_conv2

        # First fully connected layer
        self.layers['fc1'] = nn.Linear(in_features=fc_input_features, out_features=dense_units)
        
        # Second fully connected layer (output layer)
        # 10 classes for STL-10 dataset
        self.layers['fc2'] = nn.Linear(in_features=dense_units, out_features=10)

    def forward(self, x):
        # Apply the first convolutional layer
        x = self.layers['conv1'](x)
        x = self.layers['bn1'](x)
        x = F.relu(x)
        x = self.layers['pool1'](x)

        # Apply the second convolutional layer
        x = self.layers['conv2'](x)
        x = self.layers['bn2'](x)
        x = F.relu(x)
        x = self.layers['pool2'](x)

        # Apply the third convolutional layer if it exists
        if 'conv3' in self.layers:
            x = self.layers['conv3'](x)
            x = self.layers['bn3'](x)
            x = F.relu(x)
            x = self.layers['pool3'](x)

        # Adaptive pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        # Apply the fully connected layers
        x = self.layers['fc1'](x)
        x = F.relu(x)
        x = self.layers['fc2'](x)

        return x

if __name__ == "__main__":
    # You can run this as a quick test what the network is doing.
    print("0--------")
    tensor = torch.randn(3,3,92,92)
    print("00--------")
    net = CustomizableNetwork(conv_layers=2, filters_conv1=16, filters_conv2=32, filters_conv3=64, dense_units=128)
    print("1--------")
    print(net)
    out = net(tensor)
    print("2--------")
    print(out.shape)
    for name, param in net.named_parameters():
        print("3--------")
        print(name)
