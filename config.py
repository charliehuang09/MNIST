import torch
batch_size = 512
epochs = 100
learning_rate = 0.001
shuffle = True
drop_last = True
log_interval = 10
run_name = 'runs/unnamed'

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")