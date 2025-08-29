# F0 extractor (адаптация под rvc-python)
import numpy as np
import torch
import librosa

def extract_f0(wav, sr, method='crepe', device='cpu'):
    # wav: np.ndarray, float32, mono
    # sr: int
    # method: 'crepe' (по умолчанию)
    # Возвращает: f0 (np.ndarray, shape [T])
    if method == 'crepe':
        try:
            import crepe
        except ImportError:
            raise ImportError('Для F0 extraction требуется pip install crepe')
        wav16 = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav16 = wav16.astype(np.float32)
        f0, _, _ = crepe.predict(wav16, 16000, viterbi=True, step_size=10)
        f0 = f0.squeeze()
        return f0
    else:
        raise NotImplementedError(f'F0 extraction method {method} не поддерживается')
