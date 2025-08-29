
import torch
import numpy as np
import librosa
import os
from .hubert_contentvec import Hubert
from .rvc_model import RVCModel
from .f0_extractor import extract_f0
from .rmvpe_extractor import extract_f0_rmvpe
from .download_models import download_model

_rvc_cache = {}

def rvc_infer(
    wav: np.ndarray,
    sr: int,
    rvc_model_path: str,
    device: str = 'cuda',
    hubert_path: str = None,
    f0_method: str = 'crepe',
    rmvpe_model_path: str = None,
    index_path: str = None,
    index_rate: float = 0.0,
    fp16: bool = False,
    pitch_shift: int = 0,
    use_index: bool = False,
    sample_rate: int = None
) -> tuple[np.ndarray, int]:
    """
    wav: np.ndarray (float32, mono, 24kHz)
    sr: sample rate (обычно 24000)
    rvc_model_path: путь к .pth файлу модели RVC
    device: 'cuda' или 'cpu'
    hubert_path: путь к HuBERT (pth)
    f0_method: 'crepe'|'parselmouth'|'pm'|'dio' (по умолчанию crepe)
    index_path: путь к Faiss-индексу (опционально)
    index_rate: blending rate для retrieval (0.0-1.0)
    fp16: использовать half-precision
    pitch_shift: сдвиг тона (в полутоннах)
    use_index: использовать blending с индексом
    sample_rate: целевой sample rate (по умолчанию из модели)
    Возвращает: (numpy array, sample_rate)
    """
    global _rvc_cache
    cache_key = (rvc_model_path, index_path, fp16)
    if cache_key not in _rvc_cache:
        # Поиск HuBERT (автоматическая загрузка при отсутствии)
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'RVC')
        os.makedirs(models_dir, exist_ok=True)
        if hubert_path is None:
            hubert_path = os.path.join(models_dir, 'hubert_base.pt')
        if not os.path.exists(hubert_path):
            print(f"[RVC] HuBERT не найден, скачиваем автоматически...")
            hubert_path = download_model('hubert_base.pt', out_dir=models_dir)
        print(f"[RVC] Загрузка HuBERT (ContentVec): {hubert_path}")
        hubert = Hubert(hubert_path, device=device)
        print(f"[RVC] Загрузка RVC-модели: {rvc_model_path}")
        rvc = RVCModel(rvc_model_path, device=device, index_path=index_path, fp16=fp16)
        if index_path and index_rate > 0.0:
            rvc.set_index(index_path, index_rate=index_rate)
        _rvc_cache[cache_key] = (hubert, rvc)
    else:
        hubert, rvc = _rvc_cache[cache_key]

    # 1. Аудио -> float32, моно, 16kHz для HuBERT
    wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    wav16 = wav16.astype(np.float32)
    wav16 = np.expand_dims(wav16, 0)  # (1, T)
    wav16_tensor = torch.from_numpy(wav16).to(device)

    # 2. Извлечение фичей HuBERT
    units = hubert(wav16_tensor)
    # 3. Извлечение F0 (pitch)
    if f0_method == 'rmvpe':
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'RVC')
        os.makedirs(models_dir, exist_ok=True)
        if rmvpe_model_path is None:
            rmvpe_model_path = os.path.join(models_dir, 'rmvpe.pt')
        if not os.path.exists(rmvpe_model_path):
            print(f"[RVC] RMVPE не найден, скачиваем автоматически...")
            rmvpe_model_path = download_model('rmvpe.pt', out_dir=models_dir)
        f0 = extract_f0_rmvpe(wav, sr, rmvpe_model_path, device=device)
    else:
        f0 = extract_f0(wav, sr, method=f0_method, device=device)
    f0 = librosa.resample(f0, orig_sr=sr, target_sr=units.shape[1])
    f0 = np.expand_dims(f0, 0)  # (1, T)
    f0_tensor = torch.from_numpy(f0).to(device)

    # 4. Инференс RVC с поддержкой blending, fp16, pitch shift, index
    out = rvc.infer(units, f0=f0_tensor, pitch_shift=pitch_shift, use_index=use_index)
    if isinstance(out, tuple):
        wav_out = out[0]
    else:
        wav_out = out
    wav_out = wav_out.detach().cpu().numpy().squeeze()
    # 5. Привести к float32, целевой sample rate
    target_sr = sample_rate or getattr(rvc, 'sample_rate', 40000)
    wav_out = librosa.resample(wav_out, orig_sr=16000, target_sr=target_sr)
    wav_out = wav_out.astype(np.float32)
    return wav_out, target_sr
