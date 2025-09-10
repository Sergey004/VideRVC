import requests
import os

# Установите ваш API-ключ. Если переменная окружения API_KEY не установлена, будет использоваться значение по умолчанию.
API_KEY = os.getenv("API_KEY", "your_super_secret_api_key")

# URL вашего локального сервера
SERVER_URL = "http://localhost:8000/v1/audio/speech"

# Данные для отправки в запросе
payload = {
    "model": "1.5B",  # Замените на имя вашей модели
    "tokenizer_path": "Qwen/Qwen2.5-7B",
    "input": "Hello, this is a test speech.",
    "voice": "PhoneGuy_FNAF1_01"   # Замените на имя вашего голоса
}

# Заголовки запроса, включая API-ключ
headers = {
    "Content-Type": "application/json",
    "x-api-key": API_KEY
}

try:
    # Отправляем POST-запрос
    response = requests.post(SERVER_URL, json=payload, headers=headers)

    # Проверяем статус ответа
    response.raise_for_status()  # Вызовет исключение для ошибок HTTP (4xx или 5xx)

    # Сохраняем аудиофайл
    output_filename = "output_audio_with_api_key.mp3"
    with open(output_filename, "wb") as f:
        f.write(response.content)
    print(f"Аудиофайл успешно сохранен как {output_filename}")

except requests.exceptions.ConnectionError as e:
    print(f"Ошибка подключения к серверу: {e}")
    print("Убедитесь, что сервер запущен и доступен по адресу {SERVER_URL}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP ошибка: {e.response.status_code} - {e.response.text}")
except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")