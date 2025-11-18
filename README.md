# VibeVoice + RVC Speech Generation Service

## Description

Local CLI and REST API for text-to-speech generation using VibeVoice and optional processing via RVC (Voice Conversion).

---

## CLI

Example of execution:

```bash
 cli.py --text "Hello world" --reference-audio reference.wav --model-path 1.5B --out output.wav [--rvc-model rvc.pth]
```

**Parameters:**

- `--text` — text for synthesis or `--text-file` to read from file
- `--reference-audio` — reference voice (wav)
- `--model-path` — path to the VibeVoice model
- `--out` — output wav file
- `--rvc-model` — (optional) path to RVC model
- For other parameters, see `--help`

---

## REST API (FastAPI)

Starting the server:

```bash
docker build -t vibevoice-rvc .
docker run -p 8000:8000 vibevoice-rvc
```

### Endpoints

- `POST /generate` — speech generation (VibeVoice)
- `POST /convert` — WAV processing via RVC
- `POST /pipeline` — full pipeline (VibeVoice → RVC)

For request examples, see Swagger UI: <http://localhost:8000/docs>

---

## Requirements

- Python 3.10+
- CUDA (for acceleration, optional)
- ffmpeg

---
Tested on Win 11 + RTX 3090 and Ubuntu Server 24.04 LTS + RTX 3060 (4-bit Quant)

---

## Notes

- All computations are local, no internet connection required.
- To support different voices, use different reference audio and/or RVC models.
- We dont need LoRA support because we use RVC (But it will be useful for supporting other languages.)
