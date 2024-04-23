import torch
import torch.nn as nn
import numpy as np

from metaquantum.CircuitComponents import *


class VQCTorch(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, num_of_qubits=4, featuremap_depth=2, variational_depth = 2,  hadamard_gate=True, qdevice='lightning.gpu'):
        '''
        Args 
            - in_channels = Number of input embedding dimension 
            - out_channels = Number of output tensor dimension (B, out_channels=n_classes_of_labels)
            - num_of_qubits = Number of qubits. In the case of VQC, in_channels == num_of_qubits
            - variational_depth = Number of variational layers (Entangling Layers)
            - featuremap_depth = Number of Feature Embedding Layers 
            - hadamard_gate = whether to use hadamard gate for Feature Embedding Layers
            - qdevice = which interface to use. For multi-node/multi-gpu or single-node/single-gpu, this arguments should be 'lightning.gpu'
        '''
        super().__init__()
        assert in_channels == num_of_qubits
        self.vqc = VariationalQuantumBlock(in_channels=in_channels,
                                           out_channels=out_channels,
                                           num_of_qubits=num_of_qubits,
                                           featuremap_depth=featuremap_depth,
                                           variational_depth=variational_depth,
                                           hadamard_gate=hadamard_gate,
                                           qdevice=qdevice,
                                           )


    def forward(self, batch):
        output = self.vqc.forward(batch) # (B, out_channels)
        return output

