"""
FastAPI server for VibeVoice TTS generation.

Endpoints
---------
POST /synthesize
    Accepts multipart/form-data:
        - text: str (optional, can be provided via text_file)
        - text_file: UploadFile (optional, plain‑text file containing the text to synthesize)
        - reference_audio: UploadFile (required, wav or any format supported by soundfile)
        - model_path: str (required, can be a short name from model_configs.json or a full path / HF repo)
        - tokenizer_path: str (optional, required for custom models; can be omitted for short names)
        - cfg_scale: float (default 1.3)
        - temperature: float (default 0.95)
        - steps: int (default 10)
        - top_p: float (default 0.95)
        - top_k: int (default 0)
        - device: str (default "cuda")
        - do_sample: bool (default False)
        - seed: int (default 0)
        - rvc_model: str (optional, path to an RVC .pth model)
        - rvc_index: str (optional, path to an RVC Faiss index)
        - rvc_index_rate: float (optional, blending rate 0.0‑1.0)

Response
--------
StreamingResponse containing a WAV file (audio/wav) with the generated speech.
"""
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:server", host="0.0.0.0", port=8000, reload=True)
import os
import tempfile
import io
import wave
import logging
import json
import threading
from collections import OrderedDict
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request, Depends, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

# Импортируем бизнес-логику
from cli import (
    set_vibevoice_seed,
    load_vibevoice,
    vibevoice_generate,
    preprocess_reference_audio,
    download_if_hf,
    resolve_model_shortcut,
    MODEL_CONFIGS
)

# --- HuggingFace Hub ---
from huggingface_hub import snapshot_download, hf_hub_download

# --- VibeVoice imports ---
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.processor.vibevoice_tokenizer_processor import VibeVoiceTokenizerProcessor
from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

import torch
import numpy as np
import requests
import gradio as gr

# Ленивая загрузка RVC
def get_rvc_infer():
    from rvc_py.rvc_infer import rvc_infer
    return rvc_infer

# Логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VibeVoiceAPI")

# API Key for authentication
API_KEY = os.getenv("API_KEY", "your_super_secret_api_key") # Replace with a strong, unique key in production

API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key is None or api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key


# Кэш моделей VibeVoice (LRU) — чтобы не загружать модель при каждом запросе
VV_CACHE_LOCK = threading.Lock()
VV_CACHE_MAX = int(os.environ.get("VV_CACHE_MAX", "1"))  # максимум одновременных моделей в VRAM
VV_MODEL_CACHE: "OrderedDict[str, tuple]" = OrderedDict()


def get_vv_model_cached(model_path: str, tokenizer_path: Optional[str], device: str):
    """Возвращает (model, processor) из кэша или загружает и кэширует.
    LRU-кэш с ограничением по количеству моделей, чтобы не переполнить VRAM.
    """
    key = f"{os.path.abspath(model_path)}::{os.path.abspath(tokenizer_path or '')}::{device}"
    # Быстрый путь: модель уже в кэше
    with VV_CACHE_LOCK:
        if key in VV_MODEL_CACHE:
            model, processor = VV_MODEL_CACHE.pop(key)
            VV_MODEL_CACHE[key] = (model, processor)  # move to MRU
            return model, processor

    # Загрузка вне блокировки (чтобы не держать lock во время длительной операции)
    model, processor = load_vibevoice(model_path, tokenizer_path or "", device=device)

    # Регистрация в LRU-кэше и при необходимости вытеснение старых
    with VV_CACHE_LOCK:
        # Возможно, за время загрузки другой поток уже загрузил эту же модель
        if key in VV_MODEL_CACHE:
            cached_model, cached_processor = VV_MODEL_CACHE.pop(key)
            VV_MODEL_CACHE[key] = (cached_model, cached_processor)
            try:
                import torch
                # Освободим только что загруженный дубликат
                del model
                del processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Если переполнили бюджет — вытесняем LRU
        while len(VV_MODEL_CACHE) >= VV_CACHE_MAX and VV_MODEL_CACHE:
            old_key, (old_model, old_processor) = VV_MODEL_CACHE.popitem(last=False)
            try:
                import torch
                # Перевод на CPU перед удалением помогает аккуратнее освобождать VRAM
                if hasattr(old_model, "to"):
                    old_model.to("cpu")
                del old_model
                del old_processor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        VV_MODEL_CACHE[key] = (model, processor)
        return model, processor


server = FastAPI(title="VibeVoice TTS API", version="2.0.0")

# CORS (можно ограничить при необходимости)
server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Эндпоинт для чтения содержимого папки voices ---
@server.get("/voices", tags=["Files"])
async def list_voices():
    """Возвращает список файлов и конфигов в папке voices."""
    folder = os.path.join(os.getcwd(), "voices")
    if not os.path.isdir(folder):
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "voices folder not found"}})
    result = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        result.append({
            "name": name,
            "path": path,
            "type": "file" if os.path.isfile(path) else "folder"
        })
    return {"voices": result}


# --- Эндпоинт для чтения содержимого папки models/RVC ---
@server.get("/rvc-models", tags=["Files"])
async def list_rvc_models():
    """Возвращает список моделей и индексов в папке models/RVC, игнорируя *.pt файлы."""
    folder = os.path.join(os.getcwd(), "models", "RVC")
    if not os.path.isdir(folder):
        return JSONResponse(status_code=404, content={"error": {"code": "not_found", "message": "models/RVC folder not found"}})
    result = []
    for name in os.listdir(folder):
        if name.endswith(".pt"):
            continue
        path = os.path.join(folder, name)
        result.append({
            "name": name,
            "path": path,
            "type": "file" if os.path.isfile(path) else "folder"
        })
    return {"rvc_models": result}


# --- Эндпоинт со списком моделей (короткие имена из model_configs.json) ---
@server.get("/models", tags=["Files"])
async def list_models():
    try:
        keys = list(MODEL_CONFIGS.keys())
    except Exception:
        keys = []
    return {"models": sorted(keys)}

@server.get("/v1/healthcheck", tags=["Service"])
async def healthcheck():
    """Проверка работоспособности сервера."""
    return {"status": "ok"}

def resolve_text(text: Optional[str], text_file: Optional[UploadFile]) -> str:
    """Получить текст для синтеза из строки или файла."""
    if text_file:
        raw = text_file.file.read()
        try:
            text = raw.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded")
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")
    # Маркер спикера
    if not any(s in text for s in ["Speaker 1:", "Speaker 2:", "Speaker 3:", "Speaker 4:"]):
        text = f"Speaker 1: {text}"
    return text

def save_temp_file(upload: UploadFile) -> str:
    """Сохраняет загруженный файл во временное хранилище и возвращает путь."""
    suffix = os.path.splitext(upload.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        return tmp.name

def synthesize_tts(
    text: str,
    reference_audio_path: str,
    model_path: str,
    tokenizer_path: Optional[str],
    cfg_scale: float,
    temperature: float,
    steps: int,
    top_p: float,
    top_k: int,
    device: str,
    do_sample: bool,
    seed: int,
) -> tuple:
    """Генерирует речь через VibeVoice."""
    set_vibevoice_seed(seed)
    # model, processor = load_vibevoice(model_path, tokenizer_path or "", device=device)
    model, processor = get_vv_model_cached(model_path, tokenizer_path or "", device)
    wav, sr = vibevoice_generate(
        model,
        processor,
        text,
        reference_audio_path,
        cfg_scale=cfg_scale,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        device=device,
        do_sample=do_sample
    )
    return wav, sr

def postprocess_rvc(
    wav,
    sr,
    rvc_model,
    device,
    rvc_index=None,
    rvc_index_rate=None,
) -> tuple:
    """Постобработка голоса через RVC."""
    rvc_infer = get_rvc_infer()
    rvc_kwargs = {"device": device}
    if rvc_index:
        rvc_kwargs["index_path"] = rvc_index
    if rvc_index_rate is not None:
        rvc_kwargs["index_rate"] = rvc_index_rate
    import numpy as np
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = np.squeeze(wav)
    wav = wav.astype(np.float32, copy=False)
    wav, sr = rvc_infer(wav, sr, rvc_model, **rvc_kwargs)
    return wav, sr

def to_wav_bytes(wav, sr) -> bytes:
    """Преобразует numpy-массив в WAV-байты."""
    import numpy as np
    max_abs = np.max(np.abs(wav)) if np.max(np.abs(wav)) != 0 else 1.0
    wav_int16 = (wav / max_abs * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wav_int16.tobytes())
    buffer.seek(0)
    return buffer

from pydantic import BaseModel

class OpenAITTSRequest(BaseModel):
    model: str
    input: str
    tokenizer_path: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"  # "wav" или "mp3"
    speed: Optional[float] = 1.0
    device: Optional[str] = "cuda"
    seed: Optional[int] = 0
    rvc_model: Optional[str] = None
    rvc_index: Optional[str] = None
    rvc_index_rate: Optional[float] = None
    # Можно добавить другие параметры OpenAI TTS

def _process_tts_request(request: OpenAITTSRequest):
    """Общая логика генерации речи, используется UI и API эндпоинтами."""
    logger.info(f"TTS request: model={request.model}, voice={request.voice}")
    temp_files = []
    try:
        # Маппинг параметров OpenAI -> VibeVoice
        text = request.input
        model_path = request.model
        tokenizer_path = request.tokenizer_path

        # Resolve model_path and tokenizer_path from model_configs.json
        try:
            # Use resolve_model_shortcut from cli.py
            model_path, resolved_tokenizer_path = resolve_model_shortcut(model_path, tokenizer_path)
            tokenizer_path = resolved_tokenizer_path

        except FileNotFoundError:
            logger.warning("model_configs.json not found. Using model name as path.")
        except json.JSONDecodeError:
            logger.warning("Error decoding model_configs.json. Using model name as path.")

        logger.info(f"Resolved model_path: {model_path}, tokenizer_path: {tokenizer_path}")

        cfg_scale = 1.3
        temperature = 0.95
        steps = 10
        top_p = 0.95
        top_k = 0
        device = request.device or "cuda"
        do_sample = True
        seed = request.seed or 42
        rvc_model = request.rvc_model
        rvc_index = request.rvc_index
        rvc_index_rate = request.rvc_index_rate

        logger.info(f"RVC model: {rvc_model}, RVC index: {rvc_index}, RVC index rate: {rvc_index_rate}")

        # Для OpenAI TTS voice — это имя голоса из таблицы или прямой путь к эталонному аудио
        if not request.voice:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "missing_voice",
                        "message": "voice (reference audio or registered voice name) is required"
                    }
                }
            )

        voice_key = request.voice
        ref_audio_path = None
        voice_entry = None
        try:
            table = load_voice_table()
        except Exception:
            table = {}

        if isinstance(table, dict) and voice_key in table and isinstance(table[voice_key], dict):
            voice_entry = table[voice_key]
            ref_audio_path = voice_entry.get("path")
            # Наследуем RVC-параметры из записи, если они не заданы в запросе
            if not rvc_model and voice_entry.get("rvc_model"):
                rvc_model = voice_entry.get("rvc_model")
                if rvc_model.startswith("model: "):
                    rvc_model = rvc_model[len("model: "):]
            if not rvc_index and voice_entry.get("rvc_index"):
                rvc_index = voice_entry.get("rvc_index")
                if rvc_index.startswith("index: "):
                    rvc_index = rvc_index[len("index: "):]
            if rvc_index_rate is None and ("rvc_index_rate" in voice_entry):
                rvc_index_rate = voice_entry.get("rvc_index_rate")
        else:
            # Голос не найден в таблице — пробуем трактовать как путь к файлу
            ref_audio_path = voice_key

        # Проверяем существование файла эталона
        if not ref_audio_path or not os.path.isfile(ref_audio_path):
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "invalid_voice",
                        "message": f"Reference audio not found: {ref_audio_path}"
                    }
                }
            )

        text_resolved = text
        if not any(s in text_resolved for s in ["Speaker 1:", "Speaker 2:", "Speaker 3:", "Speaker 4:"]):
            text_resolved = f"Speaker 1: {text_resolved}"

        wav, sr = synthesize_tts(
            text_resolved,
            ref_audio_path,
            model_path,
            tokenizer_path,
            cfg_scale,
            temperature,
            steps,
            top_p,
            top_k,
            device,
            do_sample,
            seed,
        )
        if rvc_model:
            wav, sr = postprocess_rvc(
                wav,
                sr,
                rvc_model,
                device,
                rvc_index,
                rvc_index_rate,
            )
        buffer = to_wav_bytes(wav, sr)
        filename = f"speech.{request.response_format or 'wav'}"
        # Пока поддерживается только WAV
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException as e:
        logger.error(f"HTTPException: {e.detail}")
        return JSONResponse(status_code=e.status_code, content={"error": {"code": "http_exception", "message": e.detail}})
    except Exception as e:
        logger.exception("Unexpected error in TTS processing")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"code": "internal_error", "message": str(e)}})


@server.post("/v1/audio/speech", tags=["TTS"])
async def openai_tts(
    request: OpenAITTSRequest = Body(...),
    api_key: str = Depends(get_api_key),
):
    """OpenAI-совместимый TTS эндпоинт (требует API-ключ)."""
    return _process_tts_request(request)


# Эндпоинт для UI без авторизации
@server.post("/ui/audio/speech", tags=["TTS"])
async def openai_tts_ui(
    request: OpenAITTSRequest = Body(...),
):
    """TTS эндпоинт для UI (без API-ключа)."""
    return _process_tts_request(request)


VOICES_TABLE_PATH = os.path.join(os.getcwd(), "voices", "voices.json")


def load_voice_table() -> dict:
    if not os.path.exists(VOICES_TABLE_PATH):
        return {}
    try:
        with open(VOICES_TABLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_voice_table(table: dict) -> None:
    tmp_path = VOICES_TABLE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, VOICES_TABLE_PATH)


def auto_populate_voices():
    logger.info("Auto-populating voices from .wav.conf files...")
    voices_dir = os.path.join(os.getcwd(), "voices")
    if not os.path.isdir(voices_dir):
        logger.warning(f"Voices directory not found: {voices_dir}")
        return

    current_voice_table = load_voice_table()
    updated_voice_table = current_voice_table.copy()
    
    for filename in os.listdir(voices_dir):
        if filename.endswith(".wav.conf"):
            conf_filepath = os.path.join(voices_dir, filename)
            voice_name = filename.replace(".wav.conf", "")
            
            try:
                with open(conf_filepath, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        rvc_model = lines[0].strip()
                        rvc_index = lines[1].strip()
                        
                        # Assuming the WAV file has the same name as the conf file, but with .wav extension
                        wav_filepath = os.path.join(voices_dir, voice_name + ".wav")
                        
                        # Check if the WAV file exists, if not, skip this entry or log a warning
                        if not os.path.isfile(wav_filepath):
                            logger.warning(f"Corresponding WAV file not found for {filename}: {wav_filepath}. Skipping.")
                            continue

                        # Add or update the entry
                        updated_voice_table[voice_name] = {
                            "path": wav_filepath,
                            "rvc_model": rvc_model,
                            "rvc_index": rvc_index,
                            "rvc_index_rate": None, # Default to None, can be adjusted if needed
                            "description": f"Auto-populated from {filename}"
                        }
                        logger.info(f"Added/updated voice '{voice_name}' from {filename}")
                    else:
                        logger.warning(f"Skipping {filename}: Not enough lines in .wav.conf file.")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
    
    if updated_voice_table != current_voice_table:
        save_voice_table(updated_voice_table)
        logger.info("voices.json updated with auto-populated entries.")
    else:
        logger.info("No new voices to auto-populate or voices.json is already up-to-date.")

@server.on_event("startup")
async def startup_event():
    auto_populate_voices()


# --- Простой Gradio UI, смонтированный в /ui ---
def build_gradio_ui():
    def check_server(api_key: str):
        headers = {"x-api-key": api_key} if api_key else {}
        try:
            r = requests.get("http://localhost:8000/v1/healthcheck", headers=headers, timeout=5)
            if r.ok:
                return f"Healthcheck OK: {r.json()}"
            return f"Healthcheck failed: {r.status_code} - {r.text}"
        except Exception as e:
            return f"Healthcheck error: {e}"

    def fetch_lists(api_key: str):
        headers = {"x-api-key": api_key} if api_key else {}
        voices = []
        models = []
        try:
            r = requests.get("http://localhost:8000/voices/table", headers=headers, timeout=8)
            if r.ok:
                data = r.json()
                if isinstance(data, dict):
                    if "voices" in data and isinstance(data["voices"], dict):
                        voices = sorted(list(data["voices"].keys()))
                    else:
                        voices = sorted([k for k in data.keys()])
        except Exception:
            pass
        if not voices:
            try:
                r = requests.get("http://localhost:8000/voices", headers=headers, timeout=8)
                if r.ok:
                    data = r.json()
                    items = data.get("voices", []) if isinstance(data, dict) else []
                    voices = sorted([it.get("name") for it in items if isinstance(it, dict) and it.get("type") == "file"]) or []
            except Exception:
                pass
        try:
            r = requests.get("http://localhost:8000/models", headers=headers, timeout=5)
            if r.ok:
                data = r.json()
                models = data.get("models", []) if isinstance(data, dict) else []
        except Exception:
            pass
        return gr.update(choices=voices, value=(voices[0] if voices else None)), gr.update(choices=models, value=(models[0] if models else None))

    def synthesize(
        api_key: str,
        model: str,
        tokenizer_path: str,
        input_text: str,
        voice: str,
        device: str = "cuda",
        seed: int = 0,
        rvc_model: str | None = None,
        rvc_index: str | None = None,
        rvc_index_rate: float | None = None,
    ):
        if not input_text or not voice or not model or not tokenizer_path:
            return None, "Please fill model, tokenizer_path, voice and input text."
        # Используем UI-эндпоинт без авторизации
        url = "http://localhost:8000/ui/audio/speech"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "tokenizer_path": tokenizer_path,
            "input": input_text,
            "voice": voice,
            "device": device,
            "seed": seed,
        }
        if rvc_model:
            payload["rvc_model"] = rvc_model
        if rvc_index:
            payload["rvc_index"] = rvc_index
        if rvc_index_rate is not None:
            payload["rvc_index_rate"] = rvc_index_rate
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if not resp.ok:
                try:
                    return None, f"HTTP {resp.status_code}: {resp.json()}"
                except Exception:
                    return None, f"HTTP {resp.status_code}: {resp.text}"
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(resp.content)
                wav_path = tmp.name
            return wav_path, "Success"
        except Exception as e:
            return None, f"Request error: {e}"

    with gr.Blocks(title="VibeVoice TTS – API UI") as demo:
        gr.Markdown("""
        ### VibeVoice TTS (UI)
        This UI is served by the same FastAPI server.
        It uses a no-auth UI endpoint for synthesis.
        """)
        api_key = gr.Textbox(label="API Key (optional, for API calls)", value=os.getenv("API_KEY", ""), type="password")
        check_btn = gr.Button("Check Server")
        check_out = gr.Textbox(label="Healthcheck Result")
        with gr.Row():
            model = gr.Dropdown(label="Model", choices=[], value=None)
            tokenizer_path = gr.Textbox(label="Tokenizer Path", value="Qwen/Qwen2.5-7B")
            device = gr.Radio(choices=["cuda", "cpu"], value="cuda", label="Device")
            seed = gr.Number(label="Seed", value=0)
        input_text = gr.Textbox(label="Text", value="Hello, this is a test speech.", lines=3)
        voice = gr.Dropdown(label="Voice (from voices table)", choices=[], value=None)
        refresh_lists = gr.Button("Refresh Voices/Models")
        with gr.Accordion("Optional RVC settings", open=False):
            rvc_model = gr.Textbox(label="RVC Model (.pth)")
            rvc_index = gr.Textbox(label="RVC Index (.index)")
            rvc_index_rate = gr.Slider(label="RVC Index Rate", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
        synth_btn = gr.Button("Synthesize")
        audio_out = gr.Audio(label="Output Audio", type="filepath")
        status_out = gr.Textbox(label="Status / Errors")

        demo.load(fetch_lists, inputs=[api_key], outputs=[voice, model])
        check_btn.click(check_server, inputs=[api_key], outputs=check_out)
        refresh_lists.click(fetch_lists, inputs=[api_key], outputs=[voice, model])
        synth_btn.click(
            synthesize,
            inputs=[api_key, model, tokenizer_path, input_text, voice, device, seed, rvc_model, rvc_index, rvc_index_rate],
            outputs=[audio_out, status_out],
        )
    return demo


try:
    ui_blocks = build_gradio_ui()
    from gradio import mount_gradio_app
    server = mount_gradio_app(server, ui_blocks, path="/ui")
except Exception:
    # Если Gradio недоступен, просто пропускаем UI
    pass

from pydantic import BaseModel


class VoiceEntry(BaseModel):
    path: str  # путь к референс-аудио (wav)
    rvc_model: Optional[str] = None
    rvc_index: Optional[str] = None
    rvc_index_rate: Optional[float] = None
    description: Optional[str] = None


@server.get("/voices/table", tags=["Voices"])
async def voices_table_list():
    return load_voice_table()


@server.post("/voices/table", tags=["Voices"])
async def voices_table_upsert(name: str = Form(...), path: str = Form(...), rvc_model: Optional[str] = Form(None), rvc_index: Optional[str] = Form(None), rvc_index_rate: Optional[float] = Form(None), description: Optional[str] = Form(None)):
    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"File not found: {path}")
    table = load_voice_table()
    table[name] = VoiceEntry(path=path, rvc_model=rvc_model, rvc_index=rvc_index, rvc_index_rate=rvc_index_rate, description=description).dict()
    save_voice_table(table)
    return {"status": "ok", "message": f"Voice '{name}' added/updated."}


@server.delete("/voices/table/{name}", tags=["Voices"])
async def voices_table_delete(name: str):
    table = load_voice_table()
    if name in table:
        del table[name]
        save_voice_table(table)
        return {"status": "ok", "message": f"Voice '{name}' deleted."}
    raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")


# Эндпоинт для загрузки референс-аудио --- (для UI)
@server.post("/upload-reference-audio", tags=["Files"])
async def upload_reference_audio(file: UploadFile = File(...)):
    try:
        file_path = os.path.join("voices", file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"filename": file.filename, "path": file_path, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")


# Эндпоинт для загрузки RVC-моделей --- (для UI)
@server.post("/upload-rvc-model", tags=["Files"])
async def upload_rvc_model(file: UploadFile = File(...)):
    try:
        rvc_models_dir = os.path.join("models", "RVC")
        os.makedirs(rvc_models_dir, exist_ok=True)
        file_path = os.path.join(rvc_models_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"filename": file.filename, "path": file_path, "message": "RVC model uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload RVC model: {e}")


# Эндпоинт для загрузки RVC-индексов --- (для UI)
@server.post("/upload-rvc-index", tags=["Files"])
async def upload_rvc_index(file: UploadFile = File(...)):
    try:
        rvc_models_dir = os.path.join("models", "RVC")
        os.makedirs(rvc_models_dir, exist_ok=True)
        file_path = os.path.join(rvc_models_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        return {"filename": file.filename, "path": file_path, "message": "RVC index uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload RVC index: {e}")



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
    # models_dir is now globally defined and created, so these lines are no longer needed here.
    # os.makedirs(models_dir, exist_ok=True)

    def is_hf_repo(p):
        # repo_id или hf://repo_id
        return (p.startswith("hf://") or (not os.path.exists(p) and len(p.split("/"))==2))
    # Поддержка коротких имён
    orig_model_path, orig_tokenizer_path = model_path, tokenizer_path
    model_path, tokenizer_path = resolve_model_shortcut(model_path, tokenizer_path)
    # Model
    subfolder = None
    # Если model_path был коротким именем, ищем subfolder в конфиге
    if orig_model_path in MODEL_CONFIGS and "subfolder" in MODEL_CONFIGS[orig_model_path]:
        subfolder = MODEL_CONFIGS[orig_model_path]["subfolder"]
    if is_hf_repo(model_path):
        repo_id_full = model_path.replace("hf://", "")
        repo_id, sub_path = _split_hf_repo(repo_id_full)
        local_dir = os.path.join(models_dir, repo_id.replace("/", "__"))
        download_dir = local_dir
        if not os.path.exists(local_dir):
            print(f"Downloading VibeVoice model: {repo_id}...")
            snapshot_download(repo_id=repo_id, local_dir=local_dir)
        # Если subfolder указан в конфиге, используем его
        if subfolder:
            model_path = os.path.join(local_dir, subfolder)
        elif sub_path:
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
    # torch_dtype: bfloat16 для моделей, где это требуется
    import torch
    torch_dtype = None
    # Проверяем по имени модели или конфигу
    if ("4bit" in model_path or "bfloat16" in model_path):
        torch_dtype = torch.bfloat16
    # Можно добавить проверку по конфигу, если потребуется
    if torch_dtype:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, device_map=device, torch_dtype=torch_dtype)
    else:
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
        # Не приводим к float, если модель квантована (например, 4bit)
        is_quantized = hasattr(model, 'quantization_method') or hasattr(model, 'quantize_config') or hasattr(model, 'weight_dtype')
        if not is_quantized:
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
    wav_tensor = outputs.speech_outputs[0].detach().cpu()
    if wav_tensor.dtype == torch.bfloat16:
        wav_tensor = wav_tensor.to(torch.float32)
    wav = wav_tensor.numpy()
    return wav, 24000
