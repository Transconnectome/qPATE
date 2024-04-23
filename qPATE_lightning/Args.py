import argparse
import torch
from pytorch_lightning import Trainer
#torch.set_default_dtype(torch.float64)

#dtype = torch.DoubleTensor
#device = torch.device("cou")


def get_args():
    parser = argparse.ArgumentParser(description='Train a PATE model using hybrid quantum classical neural networks in Pytorch')
    parser = Trainer.add_argparse_args(parser)
    # arguments for global settings and computation resource 
    parser.add_argument('--n_nodes', type=int, default=1, help='number of nodes to use (default: 1)') 
    # arguments for experiment setting 
    parser.add_argument('--seed', type=int, default=42, help='seed number (default: 42)')    
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument("--data", default='MNIST', help="Dataset to train: 'MNIST', 'CatsDogs', 'Xray'")
    parser.add_argument('--val_test_together_student', action='store_true')
    parser.set_defaults(quantum=False) 
    # argumnents for qPATE-GAN   
    parser.add_argument('--quantum', action='store_true')
    parser.set_defaults(quantum=False) 
    parser.add_argument('--n_qubits', type=int, default=10, help='number of qubits to use (default: 10)')    
    parser.add_argument('--featuremap_depth', type=int, default=2, help='the depth of featuremap embedding quantum layer (default: 2)')   
    parser.add_argument('--variational_depth', type=int, default=2, help='the depth of variational quantum layer (default: 2)')  
    parser.add_argument('--entanglement', type=str, default='linear', choices=['linear', 'alternative_linear'], help="entanglement method (default:'linear')") 
    # arguments for PATE setting 
    parser.add_argument('--n_classes', type=int, default=2, help='number of label classes to predict. If GAN setting, label is (real, fake) (default: 2)')
    parser.add_argument('-n', '--n_samples', type=int, default=1000, help='number of samples to train each teacher (default: 1000)')
    parser.add_argument('--n_teachers', type=int, default=10, help='number of teachers to use (default: 10)')    
    parser.add_argument('--moments', type=float, default=0.5, help='momentum (default: 0.5) used for calculating privacy cost')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta for epsilon calculation (default: 1e-5)')
    parser.add_argument('--noise_eps', type=float, default=1e-4, help='ratio between clipping bound and std of noise applied to gradients (default: 1e-4)')
    # arguments for manifold regularization 
    parser.add_argument('--manifold_regularization', action='store_true')
    parser.set_defaults(manifold_regularization=False) 
    # arguments for training 
    parser.add_argument('--teacher_epoch', type=int, default=30, help='number of epochs to train Teachers for (default: 30)')
    parser.add_argument('--student_epoch', type=int, default=30, help='number of epochs to train student for (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-6, help='learning rate (default: 1e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=16, help='input minibatch size for training (default: 16)')
    parser.add_argument('--warmup_teacher_iter', type=int, default=1000, help='the number of warmup learning iterations for teacher discriminator')
    parser.add_argument('--num_teacher_iter', type=int, default=10, help='the number of learning iterations for teacher discriminator. If 0, only student and generator are trained')
    parser.add_argument('--num_student_iter', type=int, default=10, help='the number of learning iterations for student discriminator')
    parser.add_argument('--num_generator_iter', type=int, default=10, help='the number of warmup learning iterations for generator')
    # arguments for others
    parser.add_argument('--checkpoint', dest='useCheckpoint', action='store_const', const=True, default=False) 
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers to spawn for MP Pool (default: 4)')
    #parser.add_argument('--l2-clip', type=float, default=1., help='upper bound on the l2 norm of gradient updates (default: 0.0)')
    #parser.add_argument('--l2-penalty', type=float, default=0.001, help='l2 penalty on model weights (default: 0.001)')
    args = Trainer.parse_argparser(parser.parse_args())
    return args
    
