import numpy as np
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, noise_size):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, noise_size, 5, 1, 0, bias=False),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self, noise_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(nn.ConvTranspose2d(noise_size, 512, kernel_size = 4, stride = 1, padding = 0, bias=False),
                                   nn.BatchNorm2d(512),  
                                   nn.ReLU(True), 
                                   nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 0, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 0, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1, bias=False),
                                   nn.Tanh(),
                              )
        self._initialize_weights()
    def forward(self, x):
        x = self.model(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class Discriminator(nn.Module):
    def __init__(self, image_shape):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32,64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final = nn.Sequential(
            nn.Conv2d(256, 1, 5, 1, 0, bias=False),
            #nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, img):
        out = self.model(img)
        out = self.final(out)
        return out.view(-1, 1).squeeze(1)
    
    def inter_loss(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        loss = torch.mean((x1-x2)**2)
        return loss
    
    def inter_loss_mse(self, x1, x2):
        x1 = self.model(x1)
        x2 = self.model(x2)
        loss = torch.mean((x1-x2)**2)
        return loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

