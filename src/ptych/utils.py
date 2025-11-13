import torch

def get_default_device():
   if torch.cuda.is_available():
       return torch.device("cuda")
   if torch.backends.mps.is_available() and torch.backends.mps.is_built():
       return torch.device("mps")
   return torch.device("cpu")

def obj_to_amp(obj: torch.Tensor):
    return (torch.abs(obj) / torch.max(torch.abs(obj))).cpu()
