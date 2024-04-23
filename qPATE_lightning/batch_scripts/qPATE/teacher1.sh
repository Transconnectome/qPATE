#!/bin/bash
#SBATCH -q overrun
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --gpus-per-node=4
#SBATCH -t 1:00:00
#SBATCH -A m4138_g
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus-per-task=4

module load pytorch/2.0.1
conda init bash
source ~/.bashrc
conda activate qPATE

cd /global/homes/h/heehaw/qPATE_lightning

echo 'test 26 qubits with 10 nodes'
srun -l -u python main_PATE.py --n_nodes 2 --teacher_epoch 15 --student_epoch 25 --batch_size 256 --n_teachers 1 --n_samples 1000 --lr 1e-3 --noise_eps 1.0 --quantum --n_qubits 10 --val_test_together_student