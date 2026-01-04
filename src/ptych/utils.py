import torch

def get_default_device():
   if torch.cuda.is_available():
       return torch.device("cuda")
   if torch.backends.mps.is_available() and torch.backends.mps.is_built():
       return torch.device("mps")
   return torch.device("cpu")

def obj_to_amp(obj: torch.Tensor):
    return (torch.abs(obj) / torch.max(torch.abs(obj))).cpu()

def normalize_captures(captures: torch.Tensor, method: str = "global"):
    if method == "global":
        return (captures - captures.min()) / (captures.max() - captures.min() + 1e-8)
    if method == "each":
        return (captures - captures.min(dim=1, keepdim=True).values) / \
                          (captures.max(dim=1, keepdim=True).values - captures.min(dim=1, keepdim=True).values + 1e-8)

    raise ValueError(f"Invalid normalization method: {method}, must be 'global' or 'each'")

def check_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str | None = None):
    if not torch.all(tensor >= min_val) or not torch.all(tensor <= max_val):
        if name is not None:
            raise ValueError(f"{name} values are out of range [{min_val}, {max_val}] (found {tensor.min().item()}, {tensor.max().item()})")
        else:
            raise ValueError(f"Tensor values are out of range [{min_val}, {max_val}] (found {tensor.min().item()}, {tensor.max().item()})")
