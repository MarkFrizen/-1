import json
import csv
import re
from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
TIMEOUT_MINUTES = 60 # <-- Меняйте это число: 5, 10, 20 минут и т.д.
timeout_seconds = TIMEOUT_MINUTES * 60  # Конвертируем в секунды (библиотека openai требует секунды)

client = OpenAI(
    base_url="http://192.168.8.11:1234/v1",
    api_key="lm-studio",
    timeout=timeout_seconds  # Передаем уже в секундах
)
MODEL_NAME = "qwen/qwen3.6-27b"

def get_zero_shot_prompt(task: str) -> str:
    if task == "classify":
        return (
            "Ты классификатор тональности. Проанализируй текст и верни ТОЛЬКО валидный JSON. "
            "Никаких пояснений, никакого markdown. Только сырой JSON.\n"
            "Формат: {\"sentiment\": \"positive|negative|neutral\", \"confidence\": число от 0 до 1}"
        )
    elif task == "extract":
        return (
            "Ты извлектель сущностей. Проанализируй текст и верни ТОЛЬКО валидный JSON. "
            "Никаких пояснений, никакого markdown. Только сырой JSON.\n"
            "Формат: {\"persons\": [\"Имя Фамилия\"], \"locations\": [\"Город\"], \"organizations\": [\"Название\"]}"
        )
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def call_model(user_text: str, system_instruction: str) -> dict | None:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text}
            ],
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
            stream=False
        )

        # Исправленная работа с choices (openai v1.x+)
        choice = response.choices[0]
        content = choice.message.content
        if not content:
            print(f"[Ошибка парсинга] Пустой ответ от модели")
            return None
        
        # Пытаемся извлечь JSON из ответа (удаляем markdown-блоки и пояснения)
        json_str = content.strip()
        
        # Если ответ в markdown-блоках, извлекаем JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Пытаемся найти JSON-объект в тексте
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', json_str)
            if json_match:
                json_str = json_match.group(1)
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[Ошибка парсинга] Не удалось разобрать JSON: {e}")
            print(f"[Сырые данные] {repr(json_str)}")
            return None
    except Exception as e:
        print(f"[Критическая ошибка API] {type(e).__name__}: {e}")
        return None

def process_csv_batch(input_file: str, output_file: str, task: str):
    system_prompt = get_zero_shot_prompt(task)
    print(f"--- Начало обработки: {input_file} -> {output_file} (Задача: {task}) ---")
    print(f"(Таймаут установлен: {TIMEOUT_MINUTES} минут)")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["original_text", "result_json", "status"])
        for i, row in enumerate(reader):
            if not row or not row[0]:
                continue
            text = row[0]  # Берём первый столбец
            print(f"[{i+1}] Обработка строки... (текст: '{text[:40]}...')")
            result = call_model(text, system_prompt)
            if result:
                json_str = json.dumps(result, ensure_ascii=False)
                writer.writerow([text, json_str, "OK"])
                print(f"  -> OK: {result}")
            else:
                writer.writerow([text, "", "FAIL"])
                print("  -> FAIL")
if __name__ == "__main__":
    # Быстрый тест
    test_text = "Фильм был потрясающим, я смеялся и плакал одновременно."
    prompt = get_zero_shot_prompt("classify")
    print("--- Тестовый запуск (Zero-Shot) ---")
    result = call_model(test_text, prompt)
    if result:
        print(f"Успех! Результат: {result}")
    else:
        print("Не удалось получить ответ. Проверьте LM Studio и модель.")
        print("\nВозможные причины:")
        print("1. LM Studio не запущен или модель не загружена")
        print("2. Неправильный URL сервера (текущий: http://192.168.8.11:1234)")
        print("3. Модель не отвечает в заданном формате JSON")
        print("4. Таймаут соединения")