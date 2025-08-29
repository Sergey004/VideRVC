# VibeVoice + RVC Speech Generation Service

## Описание

Локальный CLI и REST API для генерации речи по тексту с помощью VibeVoice и опциональной обработки через RVC (Voice Conversion).

---

## CLI

Пример запуска:

```
python main.py \
  --text "Пример текста" \
  --reference-audio ref.wav \
  --model-path /path/to/vibevoice_model \
  --tokenizer-path /path/to/tokenizer.json \
  --out output.wav \
  [--rvc-model /path/to/rvc.pth]
```

**Параметры:**
- `--text` — текст для синтеза
- `--reference-audio` — эталонный голос (wav)
- `--model-path` — путь к модели VibeVoice
- `--tokenizer-path` — путь к tokenizer.json
- `--out` — выходной wav-файл
- `--rvc-model` — (опционально) путь к модели RVC
- Остальные параметры см. `--help`

---

## REST API (FastAPI)

Запуск сервера:

```
docker build -t vibevoice-rvc .
docker run -p 8000:8000 vibevoice-rvc
```

### Эндпоинты:
- `POST /generate` — генерация речи (VibeVoice)
- `POST /convert` — обработка WAV через RVC
- `POST /pipeline` — полный пайплайн (VibeVoice → RVC)

Примеры запросов см. в Swagger UI: http://localhost:8000/docs

---

## Требования
- Python 3.10+
- CUDA (для ускорения, опционально)
- ffmpeg

---

## Примечания
- Все вычисления локальные, интернет не требуется.
- Для поддержки разных голосов используйте разные reference audio и/или RVC-модели.
