from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import tempfile
import os
import uuid
import io
import base64
import urllib.request
from typing import Optional
import numpy as np
from scipy.io import wavfile
import hashlib
from pydantic import BaseModel, Field

from main import load_vibevoice, vibevoice_generate
from rvc_py.rvc_infer import rvc_infer

app = FastAPI()

# --- Global cache for models (to avoid reloading) ---
MODEL_CACHE = {}
# Health check endpoint
@app.get('/health')
def health():
    return {"status": "ok"}
# Server-side storage for reference audios
REF_STORAGE_DIR = os.path.join(os.getcwd(), "reference_storage")
os.makedirs(REF_STORAGE_DIR, exist_ok=True)

def get_vv_model(model_path, tokenizer_path, device):
    key = (model_path, tokenizer_path, device)
    if key not in MODEL_CACHE:
        MODEL_CACHE[key] = load_vibevoice(model_path, tokenizer_path, device)
    return MODEL_CACHE[key]

def get_temp_file(suffix='.wav'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path

# Helper: resolve reference_id to stored file path
def resolve_reference_id(ref_id: str):
    try:
        for name in os.listdir(REF_STORAGE_DIR):
            if name.startswith(ref_id):
                return os.path.join(REF_STORAGE_DIR, name)
    except Exception:
        pass
    return None

# Upload and persist a reference audio on the server (deterministic ID via SHA256)
@app.post('/reference/upload')
async def upload_reference(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or '')[1].lower() or '.wav'
    data = await file.read()
    # Compute deterministic ID (first 32 hex chars of SHA256)
    sha = hashlib.sha256(data).hexdigest()
    uid = sha[:32]
    save_path = os.path.join(REF_STORAGE_DIR, f"{uid}{ext}")
    if not os.path.exists(save_path):
        with open(save_path, 'wb') as f:
            f.write(data)
    return {"id": uid, "path": save_path, "sha256": sha}

# List all stored references and auto-generate IDs for existing files
@app.get('/reference/list')
def list_references():
    items = []
    try:
        for name in os.listdir(REF_STORAGE_DIR):
            path = os.path.join(REF_STORAGE_DIR, name)
            if not os.path.isfile(path):
                continue
            # Compute hash to provide a stable ID even for legacy files
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                sha = hashlib.sha256(data).hexdigest()
                uid = sha[:32]
            except Exception:
                sha = None
                uid = os.path.splitext(name)[0]
            items.append({
                "id": uid,
                "filename": name,
                "size": os.path.getsize(path),
                "path": path,
                "sha256": sha
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list references: {e}")
    return {"items": items}

# ---------------- OpenAI-like JSON TTS API ----------------
class SpeechRequest(BaseModel):
    model: str = Field(default="vibevoice", description="Model identifier, keep 'vibevoice'")
    input: str = Field(..., description="Text to speak")
    vv_model_path: str = Field(..., description="Path to VibeVoice model")
    vv_tokenizer_path: str = Field(..., description="Path to VibeVoice tokenizer")
    reference_id: Optional[str] = Field(default=None, description="Server-stored reference id")
    reference_url: Optional[str] = Field(default=None, description="URL to fetch reference audio if not using id")
    cfg_scale: float = 3.0
    temperature: float = 1.0
    steps: int = 42
    top_p: float = 1.0
    top_k: int = 0
    device: str = "cuda"
    rvc_model: Optional[str] = Field(default=None, description="Optional RVC model path for post-processing")
    response_format: str = Field(default="wav", description="wav or base64")

@app.post('/v1/audio/speech')
async def tts_v1_speech(req: SpeechRequest, background_tasks: BackgroundTasks):
    # Acquire reference audio path
    ref_path = None
    temp_paths = []
    try:
        if req.reference_id:
            ref_path = resolve_reference_id(req.reference_id)
            if not ref_path or not os.path.exists(ref_path):
                raise HTTPException(status_code=404, detail="reference_id not found")
        elif req.reference_url:
            # Download to temp
            ref_path = get_temp_file()
            urllib.request.urlretrieve(req.reference_url, ref_path)
            temp_paths.append(ref_path)
        else:
            raise HTTPException(status_code=422, detail="Provide reference_id or reference_url")

        # Load/generate
        model, processor = get_vv_model(req.vv_model_path, req.vv_tokenizer_path, req.device)
        wav, sr = vibevoice_generate(
            model, processor, req.input, ref_path,
            cfg_scale=req.cfg_scale, steps=req.steps, temperature=req.temperature,
            top_p=req.top_p, top_k=req.top_k, device=req.device
        )
        wav1d = wav.T if getattr(wav, 'ndim', 1) > 1 else wav
        if getattr(wav1d, 'ndim', 1) > 1:
            wav1d = np.squeeze(wav1d)
        wav1d = wav1d.astype(np.float32)
        out_wav = wav1d
        out_sr = sr
        if req.rvc_model:
            out_wav, out_sr = rvc_infer(wav1d, sr, req.rvc_model, device=req.device)

        if req.response_format == 'base64':
            # Encode WAV to base64 without writing file
            buf = io.BytesIO()
            wavfile.write(buf, int(out_sr), out_wav.astype(np.float32))
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            return JSONResponse({
                "model": req.model,
                "format": "wav",
                "sample_rate": int(out_sr),
                "audio": b64
            })
        # Default: return as audio/wav response file
        out_path = get_temp_file()
        temp_paths.append(out_path)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
        # schedule cleanup
        for p in temp_paths:
            background_tasks.add_task(os.remove, p)
        return FileResponse(out_path, media_type='audio/wav', filename='speech.wav')
    except HTTPException:
        # Re-raise HTTPExceptions intact
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        raise
    except Exception as e:
        # Cleanup temps and propagate error
        for p in temp_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

# ---------------------------------------------------------

@app.post('/generate')
async def generate(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_id: str | None = Form(None),
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
    # Determine source of reference audio
    ref_is_temp = False
    ref_path = None
    if reference_audio is not None:
        ref_is_temp = True
        ref_path = get_temp_file()
        with open(ref_path, 'wb') as f:
            f.write(await reference_audio.read())
    elif reference_id:
        ref_path = resolve_reference_id(reference_id)
        if not ref_path or not os.path.exists(ref_path):
            raise HTTPException(status_code=404, detail="reference_id not found")
    else:
        raise HTTPException(status_code=422, detail="Provide either reference_audio file or reference_id")

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
        if ref_is_temp and ref_path:
            try:
                os.remove(ref_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    # Schedule cleanup after response is sent
    if background_tasks is not None:
        if ref_is_temp and ref_path:
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
    # Save input to temp
    in_path = get_temp_file()
    with open(in_path, 'wb') as f:
        f.write(await audio.read())
    out_path = get_temp_file()
    try:
        sr, data = wavfile.read(in_path)
        if getattr(data, 'ndim', 1) > 1:
            data = data.T if data.shape[0] < data.shape[1] else data
            if getattr(data, 'ndim', 1) > 1:
                data = np.squeeze(data)
        data = data.astype(np.float32)
        out_wav, out_sr = rvc_infer(data, sr, rvc_model, device=device)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
    except Exception as e:
        try:
            os.remove(in_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    if background_tasks is not None:
        background_tasks.add_task(os.remove, in_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='converted.wav')

@app.post('/pipeline')
async def pipeline(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_id: str | None = Form(None),
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
    # Determine reference source
    ref_is_temp = False
    ref_path = None
    if reference_audio is not None:
        ref_is_temp = True
        ref_path = get_temp_file()
        with open(ref_path, 'wb') as f:
            f.write(await reference_audio.read())
    elif reference_id:
        ref_path = resolve_reference_id(reference_id)
        if not ref_path or not os.path.exists(ref_path):
            raise HTTPException(status_code=404, detail="reference_id not found")
    else:
        raise HTTPException(status_code=422, detail="Provide either reference_audio file or reference_id")

    in_temp = None
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
        out_wav, out_sr = rvc_infer(wav1d, sr, rvc_model, device=device)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
    except Exception as e:
        if ref_is_temp and ref_path:
            try:
                os.remove(ref_path)
            except Exception:
                pass
        if in_temp:
            try:
                os.remove(in_temp)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    if background_tasks is not None:
        if ref_is_temp and ref_path:
            background_tasks.add_task(os.remove, ref_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='pipeline.wav')

@app.post('/convert')
async def convert(
    audio: UploadFile = File(...),
    rvc_model: str = Form(...),
    device: str = Form('cuda'),
    background_tasks: BackgroundTasks = None
):
    # Save input to temp
    in_path = get_temp_file()
    with open(in_path, 'wb') as f:
        f.write(await audio.read())
    out_path = get_temp_file()
    try:
        sr, data = wavfile.read(in_path)
        if getattr(data, 'ndim', 1) > 1:
            data = data.T if data.shape[0] < data.shape[1] else data
            if getattr(data, 'ndim', 1) > 1:
                data = np.squeeze(data)
        data = data.astype(np.float32)
        out_wav, out_sr = rvc_infer(data, sr, rvc_model, device=device)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
    except Exception as e:
        try:
            os.remove(in_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    if background_tasks is not None:
        background_tasks.add_task(os.remove, in_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='converted.wav')

@app.post('/pipeline')
async def pipeline(
    text: str = Form(...),
    reference_audio: UploadFile | None = File(None),
    reference_id: str | None = Form(None),
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
    # Determine reference source
    ref_is_temp = False
    ref_path = None
    if reference_audio is not None:
        ref_is_temp = True
        ref_path = get_temp_file()
        with open(ref_path, 'wb') as f:
            f.write(await reference_audio.read())
    elif reference_id:
        ref_path = resolve_reference_id(reference_id)
        if not ref_path or not os.path.exists(ref_path):
            raise HTTPException(status_code=404, detail="reference_id not found")
    else:
        raise HTTPException(status_code=422, detail="Provide either reference_audio file or reference_id")

    in_temp = None
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
        out_wav, out_sr = rvc_infer(wav1d, sr, rvc_model, device=device)
        wavfile.write(out_path, int(out_sr), out_wav.astype(np.float32))
    except Exception as e:
        if ref_is_temp and ref_path:
            try:
                os.remove(ref_path)
            except Exception:
                pass
        if in_temp:
            try:
                os.remove(in_temp)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {e}")

    if background_tasks is not None:
        if ref_is_temp and ref_path:
            background_tasks.add_task(os.remove, ref_path)
        background_tasks.add_task(os.remove, out_path)
    return FileResponse(out_path, media_type='audio/wav', filename='final.wav')
