import torch
# Count number of devices
if torch.cuda.is_available():
    print(torch.cuda.device_count())
else:
    print(0)