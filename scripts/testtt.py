import torch
print(torch.__version__)
print(torch.backends.mps.is_available())      # True if MPS hardware detected
print(torch.backends.mps.is_built())          # Should be True as well
