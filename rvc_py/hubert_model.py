# Модуль для загрузки HuBERT (квантайзер)
# Адаптация под rvc-python (https://github.com/daswer123/rvc-python)
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class HubertSoft(nn.Module):
    def __init__(self, hubert_path, device='cpu'):
        super().__init__()
        # PyTorch >=2.6: weights_only=True по умолчанию, но HuBERT требует False
        try:
            checkpoint = torch.load(hubert_path, map_location=device, weights_only=False)
        except TypeError:
            # Для старых версий PyTorch
            checkpoint = torch.load(hubert_path, map_location=device)
        self.model = checkpoint['model']
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, wav_tensor):
        # wav_tensor: (1, T), float32, 16kHz
        with torch.no_grad():
            units = self.model.extract_features(wav_tensor)[0]
        return units
