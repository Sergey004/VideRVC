from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
import tempfile
import os
import torch
import numpy as np
from scipy.io import wavfile

from main import load_vibevoice, vibevoice_generate, rvc_convert

app = FastAPI()

# --- Global cache for models (to avoid reloading) ---
MODEL_CACHE = {}


def get_vv_model(model_path, tokenizer_path, device):
    key = (model_path, tokenizer_path, device)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_vibevoice(model_path, tokenizer_path, device)
    return MODEL_CACHE[key]


def get_temp_file(suffix='.wav'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

@app.post('/generate')
async def generate(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    cfg_scale: float = Form(3.0),
    temperature: float = Form(1.0),
    steps: int = Form(42),
    top_p: float = Form(1.0),
    top_k: int = Form(0),
    model_path: str = Form(...),
    tokenizer_path: str = Form(...),
    device: str = Form('cuda')
):
    ref_path = get_temp_file()
    with open(ref_path, 'wb') as f:
        f.write(await reference_audio.read())
    model, processor = get_vv_model(model_path, tokenizer_path, device)
    wav, sr = vibevoice_generate(
        model, processor, text, ref_path,
        cfg_scale=cfg_scale, steps=steps, temperature=temperature,
        top_p=top_p, top_k=top_k, device=device
    )
    out_path = get_temp_file()
    wavfile.write(out_path, sr, wav.T if wav.ndim > 1 else wav)
    os.remove(ref_path)
    return FileResponse(out_path, media_type='audio/wav', filename='output.wav')

@app.post('/convert')
async def convert(
    audio: UploadFile = File(...),
    rvc_model: str = Form(...),
    device: str = Form('cuda')
):
    in_path = get_temp_file()
    out_path = get_temp_file()
    with open(in_path, 'wb') as f:
        f.write(await audio.read())
    from scipy.io import wavfile
    sr, wav = wavfile.read(in_path)
    from main import get_vc
    get_vc(rvc_model, device, is_half=False)
    from rvc_infer import vc_single
    out_wav = vc_single(0, in_path, 0, None, 'rmvpe', None, 1.0)
    if isinstance(out_wav, tuple):
        out_wav = out_wav[0]
    wavfile.write(out_path, 24000, out_wav.astype(np.float32))
    os.remove(in_path)
    return FileResponse(out_path, media_type='audio/wav', filename='converted.wav')

@app.post('/pipeline')
async def pipeline(
    text: str = Form(...),
    reference_audio: UploadFile = File(...),
    cfg_scale: float = Form(3.0),
    temperature: float = Form(1.0),
    steps: int = Form(42),
    top_p: float = Form(1.0),
    top_k: int = Form(0),
    model_path: str = Form(...),
    tokenizer_path: str = Form(...),
    rvc_model: str = Form(...),
    device: str = Form('cuda')
):
    ref_path = get_temp_file()
    with open(ref_path, 'wb') as f:
        f.write(await reference_audio.read())
    model, processor = get_vv_model(model_path, tokenizer_path, device)
    wav, sr = vibevoice_generate(
        model, processor, text, ref_path,
        cfg_scale=cfg_scale, steps=steps, temperature=temperature,
        top_p=top_p, top_k=top_k, device=device
    )
    out_path = get_temp_file()
    wavfile.write(out_path, sr, wav.T if wav.ndim > 1 else wav)
    # RVC
    from main import get_vc
    get_vc(rvc_model, device, is_half=False)
    from rvc_infer import vc_single
    out_wav = vc_single(0, out_path, 0, None, 'rmvpe', None, 1.0)
    if isinstance(out_wav, tuple):
        out_wav = out_wav[0]
    wavfile.write(out_path, 24000, out_wav.astype(np.float32))
    os.remove(ref_path)
    return FileResponse(out_path, media_type='audio/wav', filename='final.wav')
