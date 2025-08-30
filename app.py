from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import os
import numpy as np
from scipy.io import wavfile

from main import load_vibevoice, vibevoice_generate
from rvc_py.rvc_infer import rvc_infer

app = FastAPI()

# --- Global cache for models (to avoid reloading) ---
MODEL_CACHE = {}
# Health check endpoint
@app.get('/health')
def health():
    return {"status": "ok"}

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
    device: str = Form('cuda'),
    background_tasks: BackgroundTasks = None
):
    ref_path = get_temp_file()
    with open(ref_path, 'wb') as f:
        f.write(await reference_audio.read())
    try:
        model, processor = get_vv_model(model_path, tokenizer_path, device)
        wav, sr = vibevoice_generate(
            model, processor, text, ref_path,
            cfg_scale=cfg_scale, steps=steps, temperature=temperature,
            top_p=top_p, top_k=top_k, device=device
        )
        out_path = get_temp_file()
        wav_to_write = wav.T if getattr(wav, 'ndim', 1) > 1 else wav
        wavfile.write(out_path, sr, wav_to_write.astype(np.float32))
    except Exception as e:
        # Cleanup temp input on failure
        try:
            os.remove(ref_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    # Schedule cleanup after response is sent
    if background_tasks is not None:
        background_tasks.add_task(os.remove, ref_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='output.wav')

@app.post('/convert')
async def convert(
    audio: UploadFile = File(...),
    rvc_model: str = Form(...),
    device: str = Form('cuda'),
    background_tasks: BackgroundTasks = None
):
    in_path = get_temp_file()
    out_path = get_temp_file()
    with open(in_path, 'wb') as f:
        f.write(await audio.read())
    try:
        sr, wav = wavfile.read(in_path)
        if getattr(wav, 'ndim', 1) > 1:
            wav = wav.mean(axis=1)
        # Convert to float32 in [-1, 1]
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        elif wav.dtype == np.uint8:
            wav = (wav.astype(np.float32) - 128.0) / 128.0
        else:
            wav = wav.astype(np.float32)
        out_wav, out_sr = rvc_infer(wav, sr, rvc_model, device=device)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
    except Exception as e:
        # Cleanup both temp files on failure
        try:
            os.remove(in_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"RVC conversion failed: {e}")
    # Schedule cleanup after response is sent
    if background_tasks is not None:
        background_tasks.add_task(os.remove, in_path)
        background_tasks.add_task(os.remove, out_path)
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
    device: str = Form('cuda'),
    background_tasks: BackgroundTasks = None
):
    ref_path = get_temp_file()
    with open(ref_path, 'wb') as f:
        f.write(await reference_audio.read())
    out_path = get_temp_file()
    try:
        model, processor = get_vv_model(model_path, tokenizer_path, device)
        wav, sr = vibevoice_generate(
            model, processor, text, ref_path,
            cfg_scale=cfg_scale, steps=steps, temperature=temperature,
            top_p=top_p, top_k=top_k, device=device
        )
        wav1d = wav.T if getattr(wav, 'ndim', 1) > 1 else wav
        if getattr(wav1d, 'ndim', 1) > 1:
            wav1d = np.squeeze(wav1d)
        wav1d = wav1d.astype(np.float32)
        rvc_wav, rvc_sr = rvc_infer(wav1d, sr, rvc_model, device=device)
        wavfile.write(out_path, int(rvc_sr), rvc_wav.astype(np.float32))
    except Exception as e:
        # Cleanup temp files on failure
        try:
            os.remove(ref_path)
        except Exception:
            pass
        try:
            os.remove(out_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")
    # Schedule cleanup after response is sent
    if background_tasks is not None:
        background_tasks.add_task(os.remove, ref_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='final.wav')
