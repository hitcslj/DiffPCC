
import torch
import torch.nn as nn 

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = nn.ConvTranspose3d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.convT2 = nn.ConvTranspose3d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.convT3 = nn.ConvTranspose3d(16, 1, 3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.convT1(x))
        x = self.relu(self.convT2(x))
        x = self.convT3(x)
        return x

class VoxelAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__ == '__main__':
    model = VoxelAE()
    x = torch.randn(1, 1, 12, 12, 12)
    out = model(x)
    print(out.shape)
     