import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule

#from VQCTorch import VQCTorch

# custom weights initialization called on the Generator
def weights_init(m):
    """
    This function initializes the model weights randomly from a 
    Normal distribution. This follows the specification from the DCGAN paper.
    https://arxiv.org/pdf/1511.06434.pdf
    Source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
      
                
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        '''
            new classical Neural Network model for classification of MNIST digits
            used for semi-supervised learning with GAN
            
            should I ever use maxpool? dropout? leaky vs not leaky ReLUs?
        '''
        super().__init__()
        hidden_dim = 32
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim , 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim ),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim, hidden_dim , 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim, hidden_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),

            nn.Conv2d(hidden_dim, hidden_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),

            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Flatten(),
            nn.Linear(4*4*hidden_dim, out_channels)
            
        )
        self.apply(weights_init)

    def forward(self, x):
        x = self.net(x)
        outputs = F.softmax(x, dim=1)
        return outputs