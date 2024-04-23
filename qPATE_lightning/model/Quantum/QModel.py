import torch.nn as nn
import torch.nn.functional as F
import torchvision
from model.Quantum.VQCTorch import VQCTorch
from model.ML.Model import weights_init

class QDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, hidden_dim=32, n_qubits=10, featuremap_depth=2, variational_depth=2, hadamard_gate=True, qdevice='lightning.gpu'):
        super().__init__()
        assert n_qubits >= out_channels

        self.net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),  # ()
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),

                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, padding=1),

                nn.Flatten(),
                nn.Linear(hidden_dim*4*4, n_qubits),
                
                # option 1 
                VQCTorch(in_channels=n_qubits, 
                         out_channels=out_channels, 
                         num_of_qubits=n_qubits, 
                         featuremap_depth=featuremap_depth, 
                         variational_depth=variational_depth, 
                         hadamard_gate=hadamard_gate, 
                         qdevice=qdevice),

                # option 2
                #VQCTorch(in_channels=n_qubits, 
                #         out_channels=n_qubits, 
                #         num_of_qubits=n_qubits, 
                #         featuremap_depth=featuremap_depth, 
                #         variational_depth=variational_depth, 
                #         hadamard_gate=hadamard_gate, 
                #         qdevice=qdevice),
                #nn.BatchNorm2d(n_qubits),
                #nn.Linear(n_qubits, out_channels),
                #nn.Softmax(dim=1)
        )
        self.apply(weights_init)
    def forward(self, input):
        return self.net(input)
