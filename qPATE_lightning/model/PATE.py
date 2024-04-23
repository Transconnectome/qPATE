import math
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from model.ML.Model import Discriminator, Generator, weights_init
from model.Quantum.QModel import QDiscriminator
from utils.helpers import pate, moments_acc

from pytorch_lightning import LightningModule






class PATE(LightningModule):
    def __init__(self, 
                 ## hyperparams for discriminators
                 # teacher
                 n_networks=1, 
                 # teacher & student
                 n_classes=2, 
                 # hyperparams for student network 
                 nz=128,
                 # hyperparams for PATE 
                 noise_eps=1e-4, 
                 num_moments = 100,
                 moments = 0.0,
                 target_delta=1e-5,
                 # hyperparams for training 
                 lr=1e-5,
                 weight_decay=1e-4,
                 # hyperparams for qPATE-GAN
                 isQuantum=False,
                 n_qubits=1, 
                 featuremap_depth=2, 
                 variational_depth=2, 
                 hadamard_gate=True, 
                 qdevice='lightning.gpu',
                 # whether to use only teacher or student
                 train_teacher=True, 
                 train_student=False,
                 teachers=None,
                 # other experimental settings 
                 val_test_together_student=False
                 ):
        # settings for torch lightning trainer manual optimization 
        super().__init__()
        #self.save_hyperparameters()
        self.automatic_optimization = False
        
        # settings for network
        self.n_networks = n_networks
        self.nz = nz 
        self.train_teacher = train_teacher
        self.train_student = train_student
        self.teachers = teachers
        if train_teacher:
            assert train_student is False 
        elif train_student: 
            assert train_teacher is False 
            assert self.teachers is not None
            self.teachers.teacher_networks.eval()
        self.val_test_together_student = val_test_together_student

        # setting for whether to use qPATE-GAN or PATE-GAN
        print('using quantum: %s' % isQuantum)
        if isQuantum:
            if self.train_teacher:
                self.teacher_networks = nn.ModuleList([QDiscriminator(in_channels=1, 
                                                            out_channels=n_classes, 
                                                            hidden_dim=32,
                                                            n_qubits=n_qubits, 
                                                            featuremap_depth=featuremap_depth,
                                                            variational_depth=variational_depth, 
                                                            hadamard_gate=hadamard_gate, 
                                                            qdevice=qdevice) for _ in range(self.n_networks)])
                
            
                self.student_networks = None
                #self.networks = nn.ModuleList([Discriminator(in_channels=1, out_channels=n_classes) for _ in range(self.n_networks)])
            elif self.train_student:
                self.teacher_networks = None 
                
                self.student_networks = QDiscriminator(in_channels=1, 
                                            out_channels=n_classes, 
                                            n_qubits=n_qubits, 
                                            hidden_dim=32,
                                            featuremap_depth=featuremap_depth,
                                            variational_depth=variational_depth, 
                                            hadamard_gate=hadamard_gate, 
                                            qdevice=qdevice)
                
                
        else:
            if self.train_teacher:
                self.teacher_networks = nn.ModuleList([Discriminator(in_channels=1, out_channels=n_classes) for _ in range(self.n_networks)])
                self.student_networks = None
            elif self.train_student:
                self.teacher_networks = None 
                self.student_networks = Discriminator(in_channels=1, out_channels=n_classes)
        
        
        # settings for loss 
        self.criterion = nn.CrossEntropyLoss()

        # settings for PATE
        self.noise_eps = noise_eps
        self.alpha = torch.tensor([moments for _ in range(num_moments)], device=self.device)
        self.l_list = 1 + torch.tensor(range(num_moments))
        self.target_delta = target_delta 
        
        # settings for Training
        self.lr = lr
        self.weight_decay = weight_decay



    def training_step(self, batch, batch_idx):         
        # Training loop for teacher network 
        if self.train_teacher:
            teacher_optimizers = self.optimizers()
            student_optimizers = None 
            if not isinstance(teacher_optimizers, list): 
                teacher_optimizers = [teacher_optimizers]
            batch_size = batch['teacher0'][0].size()[0]
            teacher_loss = []
            teacher_acc = []
            for ith in range(self.n_networks):
                teacher_optimizers[ith].zero_grad()
                img, targets = batch['teacher%s' % ith] 
                outputs = self.teacher_networks[ith](img)
                loss = self.criterion(outputs, targets.to(outputs.device))
                self.manual_backward(loss)
                self.clip_gradients(teacher_optimizers[ith], gradient_clip_val=5, gradient_clip_algorithm="norm")
                teacher_optimizers[ith].step()
                
                ## logging each teachers performance
                teacher_loss.append(loss.item())
                _, predicts = torch.max(outputs.detach(), 1)
                corrects = (predicts == targets.to(outputs.device)).sum().item() 
                teacher_acc.append(100 * corrects / batch_size)

            #### calculating he current privacy cost and summarizing 
            self.log_dict({'train_teacher_loss': torch.tensor(teacher_loss).mean().item(), 'train_teacher_acc': torch.tensor(teacher_acc).mean().item()})

        elif self.train_student: 
            teacher_optimizers = None
            student_optimizers = self.optimizers()
            student_optimizers.zero_grad() 
            img, targets = batch 
            batch_size = img.size()[0]
            outputs = self.student_networks(img)
            predictions, _ = pate(self.teachers.teacher_networks, img.detach(), lap_scale=self.noise_eps)
            loss = self.criterion(outputs, predictions.to(outputs.device))
            self.manual_backward(loss)
            self.clip_gradients(student_optimizers, gradient_clip_val=5, gradient_clip_algorithm="norm")
            student_optimizers.step() 
        
            ## update the moments
            _, predicts = torch.max(outputs.detach(), 1)
            corrects = (predicts == targets.to(outputs.device)).sum().item() 
            student_acc = (100 * corrects / batch_size)
            self.log_dict({'train_student_loss': loss.item(), 'train_student_acc': student_acc})
  


    def validation_step(self, batch, batch_idx): 
        if self.train_teacher:
            batch_size = batch['teacher0'][0].size()[0]
            teacher_loss = []
            teacher_acc = []
            for ith in range(self.n_networks):
                img, targets = batch['teacher%s' % ith] 
                outputs = self.teacher_networks[ith](img)
                loss = self.criterion(outputs, targets.to(outputs.device))
                
                ## logging each teachers performance
                teacher_loss.append(loss.item())
                _, predicts = torch.max(outputs.detach(), 1)
                corrects = (predicts == targets.to(outputs.device)).sum().item() 
                teacher_acc.append(100 * corrects / batch_size)
                
                #### calculating he current privacy cost and summarizing 
                self.log_dict({'val_teacher_loss': torch.tensor(teacher_loss).mean().item(), 'val_teacher_acc': torch.tensor(teacher_acc).mean().item()})

        elif self.train_student: 
            self.teachers.teacher_networks.eval()
            if self.val_test_together_student is False: 
                img, targets = batch 
                batch_size = img.size()[0]
                outputs = self.student_networks(img)
                predictions, _ = pate(self.teachers.teacher_networks, img.detach(), lap_scale=self.noise_eps)
                loss = self.criterion(outputs, predictions.to(outputs.device))
                _, predicts = torch.max(outputs.detach(), 1)
                corrects = (predicts == targets.to(outputs.device)).sum().item() 
                student_acc = (100 * corrects / batch_size)
                self.log_dict({'val_student_loss': loss.item(), 'val_student_acc': student_acc})
            else: 
                val_img, val_targets = batch['val']
                test_img, test_targets = batch['test']
                assert val_img.size()[0] == test_img.size()[0]
                batch_size = val_img.size()[0] 
                # validation step 
                val_outputs = self.student_networks(val_img)
                val_predictions, _ = pate(self.teachers.teacher_networks, val_img.detach(), lap_scale=self.noise_eps)
                val_loss = self.criterion(val_outputs, val_predictions.to(val_outputs.device))
                _, val_predicts = torch.max(val_outputs.detach(), 1)
                val_corrects = (val_predicts == val_targets.to(val_outputs.device)).sum().item() 
                val_student_acc = (100 * val_corrects / batch_size)
                self.log_dict({'val_student_loss': val_loss.item(), 'val_student_acc': val_student_acc})                
                # test step 
                test_outputs = self.student_networks(test_img)
                test_predictions, _ = pate(self.teachers.teacher_networks, test_img.detach(), lap_scale=self.noise_eps)
                test_loss = self.criterion(test_outputs, test_predictions.to(test_outputs.device))
                _, test_predicts = torch.max(test_outputs.detach(), 1)
                test_corrects = (test_predicts == test_targets.to(test_outputs.device)).sum().item() 
                test_student_acc = (100 * test_corrects / batch_size)
                self.log_dict({'test_student_loss': test_loss.item(), 'test_student_acc': test_student_acc})                   

        

    def test_step(self, batch, batch_idx): 
        if self.train_teacher:
            batch_size = batch['teacher0'][0].size()[0]
            teacher_loss = []
            teacher_acc = []
            for ith in range(self.n_networks):
                img, targets = batch['teacher%s' % ith] 
                outputs = self.teacher_networks[ith](img)
                loss = self.criterion(outputs, targets.to(outputs.device))
                
                ## logging each teachers performance
                teacher_loss.append(loss.item())
                _, predicts = torch.max(outputs.detach(), 1)
                corrects = (predicts == targets.to(outputs.device)).sum().item()  
                teacher_acc.append(100 * corrects / batch_size)
                
                #### calculating he current privacy cost and summarizing 
                self.log_dict({'test_teacher_loss': torch.tensor(teacher_loss).mean().item(), 'test_teacher_acc': torch.tensor(teacher_acc).mean().item()})

        elif self.train_student: 
            img, targets = batch 
            batch_size = img.size()[0]
            outputs = self.student_networks(img)
            predictions, _ = pate(self.teachers.teacher_networks, img.detach(), lap_scale=self.noise_eps)
            loss = self.criterion(outputs, predictions.to(outputs.device))
        
            ## update the moments
            _, predicts = torch.max(outputs.detach(), 1)
            corrects = (predicts == targets.to(outputs.device)).sum().item() 
            student_acc = (100 * corrects / batch_size)
            self.log_dict({'test_student_loss': loss.item(), 'test_student_acc': student_acc})
        
    

    def configure_optimizers(self):
        if self.train_teacher: 
            return [optim.AdamW(self.teacher_networks[ith].parameters(), lr=self.lr, weight_decay=self.weight_decay) for ith in range(self.n_networks)]
        elif self.train_student: 
            return optim.AdamW(self.student_networks.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    