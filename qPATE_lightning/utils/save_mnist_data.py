import torch

mnist_path = "data/MNIST/"              # for loading and saving tensors
try:
    from metaquantum.Utils.Dataset import mnist
    #x_train, y_train, x_test, y_test = mnist.data_loading_two_target(target_1 = 2, target_2 = 3, padding = True)
    # WHY was I having it pull the targets 2 and 3!!!!!!!!!!!!
    x_train, y_train, x_test, y_test = mnist.data_loading_two_target(target_1 = 0, target_2 = 1, padding = True)
    
    print(f"length of array {x_train.size()[0]}")
    print(f"length of array {x_test.size()[0]}")
    
    x_train = x_train.reshape((x_train.size()[0], 1, 32, 32))
    x_test = x_test.reshape((x_test.size()[0], 1, 32, 32))

    print("Saving samples tensors to file...")
    torch.save(x_train, mnist_path + 'x_train.pt', _use_new_zipfile_serialization=False)
    torch.save(x_test, mnist_path + 'x_test.pt', _use_new_zipfile_serialization=False)
    torch.save(y_train, mnist_path + 'y_train.pt', _use_new_zipfile_serialization=False)
    torch.save(y_test, mnist_path + 'y_test.pt', _use_new_zipfile_serialization=False)
    
except ImportError:
    print("Error importing MNIST from metaquantum!")
    try:
        print("loading preset samples from torch files...")
        print("device:", args.device)
        x_train = torch.load(mnist_path + 'x_train.pt', map_location=args.device)
        x_test = torch.load(mnist_path + 'x_test.pt', map_location=args.device)
        y_train = torch.load(mnist_path + 'y_train.pt', map_location=args.device)
        y_test = torch.load(mnist_path + 'y_test.pt', map_location=args.device)  
    except:
        print("Error loading preset samples!")
   