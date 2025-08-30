# rvc_py

Минималистичная реализация RVC (Retrieval-based Voice Conversion) для Python 3.11, адаптированная из https://github.com/daswer123/rvc-python.

- Совместимо с основным проектом VibeVoice.
- Не требует сторонних UI.
- Для интеграции используйте функцию rvc_infer из main.py.

## Особенности инференса (адаптировано из WebUI RVC)

- **Поддержка sample rate:** 32kHz, 40kHz, 48kHz (определяется по модели, параметр sr).
- **Параметры инференса:** pitch shift, index rate (retrieval blending), f0 method (rmvpe/torchcrepe/parselmouth/pm/dio), hop size, block size, auto predict f0, fp16/half-precision.
- **Векторный поиск:** поддержка Faiss для retrieval-based voice conversion (ускоряет и улучшает качество).
- **Гибкая обработка аудио:** автоматическое определение sample rate, hop size, block size.
- **Архитектура:** модульная — отдельные модули для препроцессинга, извлечения фичей, инференса, поиска.
- **Зависимости:** torch, faiss, librosa, soundfile, numpy, torchcrepe, parselmouth, onnxruntime, pyworld, scipy.

## Установка зависимостей

```
pip install torch numpy librosa soundfile scipy torchcrepe faiss-cpu parselmouth onnxruntime pyworld
```

## Использование

Импортируйте rvc_infer из rvc_py/rvc_infer.py и вызывайте как post-process.

---

Исходный репозиторий: https://github.com/daswer123/rvc-python
WebUI: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Адаптация: Python 3.11, без UI, только инференс.
