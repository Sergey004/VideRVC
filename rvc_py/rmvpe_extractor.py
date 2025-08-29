# RMVPE pitch extractor (адаптация под WebUI RVC)
# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main/rmvpe
import numpy as np
import torch
import os

class RMVPE:
    def __init__(self, model_path, device='cpu'):
        # Здесь должна быть ваша логика загрузки модели RMVPE
        # Например, torch.load(model_path)
        self.device = device
        self.model_path = model_path
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)

    def infer(self, wav, sr):
        # wav: np.ndarray, float32, mono
        # sr: sample rate
        # Здесь должна быть ваша логика инференса RMVPE
        # Ниже — заглушка (возвращает нули)
        length = wav.shape[0]
        f0 = np.zeros(length, dtype=np.float32)
        return f0

def extract_f0_rmvpe(wav, sr, rmvpe_model_path, device='cpu'):
    rmvpe = RMVPE(rmvpe_model_path, device=device)
    return rmvpe.infer(wav, sr)
