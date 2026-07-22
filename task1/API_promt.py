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
            max_tokens=1024,
            stream=False
        )

        # Исправленная работа с choices (openai v1.x+)
        choice = response.choices[0]
        content = choice.message.content
        
        if not content:
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
        except json.JSONDecodeError:
            return None
    except Exception as e:
        return None

def process_csv_batch(input_file: str, output_file: str, task: str):
    system_prompt = get_zero_shot_prompt(task)
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        next(reader)  # Пропускаем заголовок
        for i, row in enumerate(reader):
            if not row or not row[0]:
                continue
            text = row[0]  # Берём первый столбец
            result = call_model(text, system_prompt)
            
            entry = {
                "id": i + 1,
                "text": text,
                "prediction": result
            }
            
            results.append(entry)
    
    # Сохраняем в JSON файл
    output_data = {
        "task": task,
        "model": MODEL_NAME,
        "total_count": len(results),
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Обработка CSV и сохранение в JSON
    process_csv_batch("input.csv", "output.json", "classify")
