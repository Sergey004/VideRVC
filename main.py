
# --- Seed utils ---
import random


def set_vibevoice_seed(seed: int):
    """Sets the seed for torch, numpy, and random, handling large seeds for numpy."""
    if seed == 0:
        seed = random.randint(1, 0xffffffffffffffff)
    MAX_NUMPY_SEED = 2**32 - 1
    numpy_seed = seed % MAX_NUMPY_SEED
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(numpy_seed)
    random.seed(seed)
import soundfile as sf
import librosa

# Аналог функции из vibevoice_nodes.py
def preprocess_reference_audio(audio_path, target_sr=24000):
    import numpy as np
    audio, sr = sf.read(audio_path)
    print(f"[DEBUG] Reference audio: {audio_path}")
    print(f"[DEBUG]  - original shape: {audio.shape}")
    print(f"[DEBUG]  - original dtype: {audio.dtype}")
    print(f"[DEBUG]  - original sample rate: {sr}")
    print(f"[DEBUG]  - duration: {audio.shape[0] / sr:.2f} sec")
    print(f"[DEBUG]  - min: {audio.min():.4f}, max: {audio.max():.4f}")
    # Convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        print(f"[DEBUG]  - converted to mono, new shape: {audio.shape}")
    # Remove NaN/Inf
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        print("[DEBUG]  - contains NaN/Inf, replacing with zeros")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize extreme values
    max_val = np.abs(audio).max()
    if max_val > 10.0:
        print(f"[DEBUG]  - values very large (max: {max_val}), normalizing")
        audio = audio / max_val
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        print(f"[DEBUG]  - resampled to {target_sr} Hz, new shape: {audio.shape}")
        sr = target_sr
    # Final check after resampling
    if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
        print("[DEBUG]  - contains NaN/Inf after resampling, replacing with zeros")
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    # Convert to float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        print(f"[DEBUG]  - converted to float32")
    # Normalize if values outside [-1, 1]
    max_abs = np.max(np.abs(audio))
    if max_abs > 1.01:
        audio = audio / max_abs
        print(f"[DEBUG]  - normalized to [-1, 1] range")
    print(f"[DEBUG]  - final shape: {audio.shape}")
    print(f"[DEBUG]  - final dtype: {audio.dtype}")
    print(f"[DEBUG]  - final sample rate: {sr}")
    print(f"[DEBUG]  - final min: {audio.min():.4f}, max: {audio.max():.4f}")
    return audio, sr

import argparse
import torch
import numpy as np
from scipy.io import wavfile
import os


# --- HuggingFace Hub ---
from huggingface_hub import snapshot_download, hf_hub_download

# --- Model configs (короткие имена) ---
import json
import os

_MODEL_CONFIGS_PATH = os.path.join(os.path.dirname(__file__), "model_configs.json")
with open(_MODEL_CONFIGS_PATH, "r", encoding="utf-8") as _f:
    MODEL_CONFIGS = json.load(_f)


# --- VibeVoice imports ---
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast





def resolve_model_shortcut(model_path, tokenizer_path):
    """Если model_path или tokenizer_path — короткое или человекочитаемое имя (1.5B, 7B, VibeVoice-1.5B, VibeVoice-Large-pt), подставить repo_id и tokenizer_repo."""
    if model_path in MODEL_CONFIGS:
        model_path = MODEL_CONFIGS[model_path]["repo_id"]
    if tokenizer_path in MODEL_CONFIGS:
        tokenizer_path = MODEL_CONFIGS[tokenizer_path]["tokenizer_repo"]
    return model_path, tokenizer_path


def _split_hf_repo(path: str):
    """
    Split a string that may contain a sub‑folder.
    Example:
        "username/repo/subdir" → ("username/repo", "subdir")
        "username/repo"        → ("username/repo", "")
    """
    parts = path.split("/")
    if len(parts) > 2:
        repo_id = "/".join(parts[:2])
        sub_path = "/".join(parts[2:])
    else:
        repo_id = path
        sub_path = ""
    return repo_id, sub_path


def download_if_hf(model_path, tokenizer_path, models_dir="models"):
    """Если путь похож на huggingface repo (например, repo_id или repo_id:path), скачать в models_dir. Возвращает локальные пути к model_path и tokenizer_path."""
    os.makedirs(models_dir, exist_ok=True)
    def is_hf_repo(p):
        # repo_id или hf://repo_id
        return (p.startswith("hf://") or (not os.path.exists(p) and len(p.split("/"))==2))
    # Поддержка коротких имён
    model_path, tokenizer_path = resolve_model_shortcut(model_path, tokenizer_path)
    # Model
    if is_hf_repo(model_path):
        repo_id_full = model_path.replace("hf://", "")
        repo_id, sub_path = _split_hf_repo(repo_id_full)
        local_dir = os.path.join(models_dir, repo_id.replace("/", "__"))
        download_dir = local_dir
        if not os.path.exists(local_dir):
            print(f"Downloading VibeVoice model: {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        if sub_path:
            model_path = os.path.join(local_dir, sub_path)
        else:
            model_path = local_dir
        index_file = os.path.join(model_path, "model.safetensors.index.json")
    # Tokenizer
    if is_hf_repo(tokenizer_path):
        repo_id = tokenizer_path.replace("hf://", "")
        local_tokenizer = os.path.join(models_dir, repo_id.replace("/", "__")+"_tokenizer.json")
        if not os.path.exists(local_tokenizer):
            print(f"Downloading tokenizer.json for {repo_id}...")
            tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=models_dir, local_dir_use_symlinks=False)
            os.rename(tokenizer_file_path, local_tokenizer)
        tokenizer_path = local_tokenizer
    return model_path, tokenizer_path

def load_vibevoice(model_path, tokenizer_path, device='cuda'):
    model_path, tokenizer_path = download_if_hf(model_path, tokenizer_path)
    tokenizer = VibeVoiceTextTokenizerFast(tokenizer_file=tokenizer_path)
    audio_processor = VibeVoiceTokenizerProcessor()
    processor = VibeVoiceProcessor(tokenizer=tokenizer, audio_processor=audio_processor)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, device_map=device)
    model.eval()
    return model, processor


def vibevoice_generate(model, processor, text, reference_audio, cfg_scale=1.3, steps=10, temperature=0.95, top_p=0.95, top_k=0, device='cuda', do_sample=True):
    # Reference audio: path to wav file
    ref_audio, sr = preprocess_reference_audio(reference_audio, target_sr=24000)
    # Prepare input for processor
    inputs = processor(
        text=[text],
        voice_samples=[[ref_audio]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True
    )
    # Проверка на NaN/Inf
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if torch.any(torch.isnan(value)) or torch.any(torch.isinf(value)):
                print(f"[ERROR] Input tensor '{key}' contains NaN or Inf values")
                raise ValueError(f"Invalid values in input tensor: {key}")
    # Перенос на устройство
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    # Seed (опционально)
    # set_vibevoice_seed(seed) # Вызывать из main
    model.set_ddpm_inference_steps(num_steps=steps)
    generation_config = {'do_sample': do_sample, 'temperature': temperature, 'top_p': top_p}
    if top_k > 0:
        generation_config['top_k'] = top_k
    # Аппаратные оптимизации для eager
    if device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.empty_cache()
        model = model.float()
        processed_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype in [torch.int, torch.long, torch.int32, torch.int64, torch.bool, torch.uint8]:
                    processed_inputs[k] = v
                elif "mask" in k.lower():
                    processed_inputs[k] = v.bool() if v.dtype != torch.bool else v
                else:
                    processed_inputs[k] = v.float()
            else:
                processed_inputs[k] = v
        inputs = processed_inputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer, generation_config=generation_config,
            verbose=False
        )
    wav = outputs.speech_outputs[0].detach().cpu().numpy()
    return wav, 24000


def main():
    parser = argparse.ArgumentParser(description='VibeVoice CLI')
    parser.add_argument('--text', type=str, required=False, help='Text to synthesize')
    parser.add_argument('--text-file', type=str, required=False, help='Path to text file for TTS')
    parser.add_argument('--reference-audio', type=str, required=True, help='Reference audio wav')
    parser.add_argument('--cfg-scale', type=float, default=1.3, help='Classifier-Free Guidance scale (default: 1.3)')
    parser.add_argument('--temperature', type=float, default=0.95, help='Sampling temperature (default: 0.95)')
    parser.add_argument('--steps', type=int, default=10, help='Number of diffusion steps (default: 10)')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p sampling (default: 0.95)')
    parser.add_argument('--top-k', type=int, default=0, help='Top-k sampling (default: 0)')
    parser.add_argument('--rvc-model', type=str, default=None, help='Path to RVC model (.pth)')
    parser.add_argument('--rvc-index', type=str, default=None, help='Path to RVC Faiss index (.index)')
    parser.add_argument('--rvc-index-rate', type=float, default=0.0, help='Retrieval blending rate (0.0-1.0)')
    parser.add_argument('--out', type=str, required=True, help='Output wav file')
    parser.add_argument('--model-path', type=str, required=True, help='Path to VibeVoice model (or short name)')
    parser.add_argument('--tokenizer-path', type=str, required=False, help='Path to VibeVoice tokenizer.json (or short name, default is chosen by model)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for computation')
    parser.add_argument('--do_sample', action='store_true', help='Включить sampling (по умолчанию False)')
    parser.add_argument('--seed', type=int, default=0, help='Seed для генерации (0 = случайный)')
    args = parser.parse_args()

    # Получить текст: либо из файла, либо из аргумента
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            raw_text = f.read()
        # Convert multi-line text to a single line with spaces between sentences
        text = ' '.join(raw_text.split())
    elif args.text:
        text = args.text.strip()
    else:
        raise ValueError("Укажите --text или --text-file!")

    # Если не указан --tokenizer-path, подставить по имени модели
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path:
        # Если model-path — короткое или человекочитаемое имя, взять из MODEL_CONFIGS
        if args.model_path in MODEL_CONFIGS:
            tokenizer_path = MODEL_CONFIGS[args.model_path]["tokenizer_repo"]
        else:
            raise ValueError("--tokenizer-path обязателен для кастомных моделей!")

    # Если текст не содержит 'Speaker', добавить автоматически для одного говорящего
    if not any(s in text for s in ["Speaker 1:", "Speaker 2:", "Speaker 3:", "Speaker 4:"]):
        text = f"Speaker 1: {text}"
    # Seed
    set_vibevoice_seed(args.seed)
    model, processor = load_vibevoice(args.model_path, tokenizer_path, device=args.device)
    wav, sr = vibevoice_generate(
        model, processor, text, args.reference_audio,
        cfg_scale=args.cfg_scale, steps=args.steps, temperature=args.temperature,
        top_p=args.top_p, top_k=args.top_k, device=args.device,
        do_sample=args.do_sample
    )
    # Post-process через RVC, если указан путь к модели
    from rvc_py.rvc_infer import rvc_infer
    if args.rvc_model:
        print(f"[INFO] Post-process через RVC: {args.rvc_model}")
        rvc_kwargs = dict(device=args.device)
        if args.rvc_index:
            rvc_kwargs['index_path'] = args.rvc_index
        if args.rvc_index_rate:
            rvc_kwargs['index_rate'] = args.rvc_index_rate
        # Ensure waveform is 1D float32 numpy array for RVC/ RMVPE
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().cpu().numpy()
        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = np.squeeze(wav)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        wav = wav.astype(np.float32, copy=False)
        wav, sr = rvc_infer(wav, sr, args.rvc_model, **rvc_kwargs)
    # Сохраняем результат
    print(f"[INFO] Saved output to {args.out}")
    wavfile.write(args.out, sr, wav.T if wav.ndim > 1 else wav)

if __name__ == '__main__':
    main()
