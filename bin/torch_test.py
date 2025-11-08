import torch, torchaudio, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("CUDA dostępne:", torch.cuda.is_available())
