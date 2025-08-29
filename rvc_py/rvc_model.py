# RVC Model (адаптация под rvc-python)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
try:
    import faiss
except ImportError:
    faiss = None
    print("[WARN] faiss не установлен — retrieval blending будет недоступен.")

class RVCModel(nn.Module):
    def __init__(self, model_path, device='cpu', index_path=None, fp16=False, sample_rate=None):
        super().__init__()
        checkpoint = torch.load(model_path, map_location=device)
        if 'model' in checkpoint:
            self.model = checkpoint['model']
        else:
            self.model = checkpoint
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.fp16 = fp16
        self.sample_rate = sample_rate or self._get_sample_rate_from_config(checkpoint)
        # Retrieval index (Faiss)
        self.index = None
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        self.index_rate = 0.0

    def _get_sample_rate_from_config(self, checkpoint):
        # Попытка определить sample rate из конфига модели
        if 'config' in checkpoint and 'sr' in checkpoint['config']:
            return int(checkpoint['config']['sr'])
        return 40000

    def set_index(self, index_path, index_rate=0.5):
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.index_rate = index_rate

    def infer(self, units, f0=None, pitch_shift=0, use_index=False):
        # units: (1, T, C)
        # f0: (1, T) or None
        # pitch_shift: int (сдвиг тона)
        # use_index: bool (использовать retrieval blending)
        with torch.no_grad():
            # Pitch shift (если требуется)
            if pitch_shift != 0 and f0 is not None:
                f0 = f0 * (2 ** (pitch_shift / 12))
            # FP16
            if self.fp16:
                units = units.half()
                if f0 is not None:
                    f0 = f0.half()
            # Retrieval blending (Faiss)
            if use_index and self.index is not None and self.index_rate > 0.0:
                # Пример: смешивание с ближайшими векторами из индекса
                units_np = units.cpu().numpy().squeeze()
                D, I = self.index.search(units_np, 1)
                retrieved = self.index.reconstruct_n(0, units_np.shape[0])
                units_blend = (1 - self.index_rate) * units_np + self.index_rate * retrieved
                units = torch.from_numpy(units_blend).unsqueeze(0).to(self.device)
            # Инференс
            if f0 is not None:
                output = self.model(units, f0)
            else:
                output = self.model(units)
        return output
