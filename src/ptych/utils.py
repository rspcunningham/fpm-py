import torch

def get_default_device():
   if torch.cuda.is_available():
       return torch.device("cuda")
   if torch.backends.mps.is_available() and torch.backends.mps.is_built():
       return torch.device("mps")
   return torch.device("cpu")
