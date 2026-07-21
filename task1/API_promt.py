import json
import csv
from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
client = OpenAI(
    base_url="http://192.168.8.11:1234/v1",
    api_key="lm-studio",
    timeout=300  # Увеличили до 5 минут — критично для больших моделей
)

# Для тестов лучше сразу поставить маленькую модель: qwen2.5-7b
# Если оставите 35B — будет очень медленно и часто таймауты
MODEL_NAME = "qwen2.5-7b"  # <-- замените на эту модель для стабильных тестов
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
            temperature=0.0,   # Детерминированный вывод
            top_p=1.0,
            max_tokens=256,
            stream=False
        )

        # Исправленная работа с choices (openai v1.x+)
        choice = response.choices[0]
        content = choice.message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"[Ошибка парсинга] Модель вернула не JSON:\n{content}")
            return None
    except Exception as e:
        print(f"[Критическая ошибка API] {type(e).__name__}: {e}")
        return None
def process_csv_batch(input_file: str, output_file: str, task: str):
    system_prompt = get_zero_shot_prompt(task)
    print(f"--- Начало обработки: {input_file} -> {output_file} (Задача: {task}) ---")
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(["original_text", "result_json", "status"])
        for i, row in enumerate(reader):
            if not row or not row[0]:
                continue
            text = row[0]  # <-- исправлено: берём первый столбец, а не всю строку
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