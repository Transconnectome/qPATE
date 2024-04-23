# qPATE


# example 
### runnning classical PATE
```
python main_PATE.py --teacher_epoch 35 --student_epoch 35 --batch_size 256 --n_teachers 12 --n_samples 1000 --lr 1e-3 --noise_eps 1
```
  
### running quantum PATE
```
python main_PATE.py --teacher_epoch 35 --student_epoch 35 --batch_size 256 --n_teachers 12 --n_samples 1000 --lr 1e-3 --noise_eps 1 --quantum --n_qubits 2
```