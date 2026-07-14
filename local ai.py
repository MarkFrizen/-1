import csv
import json
from openai import OpenAI

# ================== НАСТРОЙКИ ==================
BASE_URL = "http://192.168.0.140:1234/v1"
API_KEY = "not-needed"          # если аутентификация отключена
MODEL = "qwen/qwen3.6-35b-a3b"
TEMPERATURE = 0.5               # экспериментируйте: 0.0 – строго, 1.0 – креативно
TOP_P = 0.9
TASK = "summarize"              # можно заменить на "extract_entities" или "classify"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output_local_ai.csv"
# ==============================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Чтение CSV с автоподбором кодировки
encodings = ["utf-8-sig", "cp1251", "utf-8"]
texts = []
for enc in encodings:
    try:
        with open(INPUT_FILE, "r", encoding=enc) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text") and row["text"].strip():
                    texts.append(row["text"])
        print(f"✅ Файл прочитан в кодировке {enc}")
        break
    except (UnicodeDecodeError, KeyError):
        continue
else:
    print("❌ Не удалось прочитать файл.")
    exit(1)

# Промпты в зависимости от задачи
if TASK == "summarize":
    SYSTEM_PROMPT = """
Ты — ассистент, который кратко пересказывает тексты.

Примеры (Few-shot):
Текст: "Вчера компания X объявила о рекордной прибыли."
Пересказ: "Компания X сообщила о рекордной прибыли."

Текст: "Учёные открыли новый вид динозавра в Аргентине."
Пересказ: "В Аргентине обнаружен новый вид динозавра."

Теперь перескажи следующий текст. Сначала выдели ключевые факты (Chain-of-Thought), затем дай краткий пересказ в формате JSON:
{
  "summary": "краткий пересказ",
  "keywords": ["ключевое1", "ключевое2"]
}
"""
    USER_TEMPLATE = "Текст для пересказа:\n\n{text}"

elif TASK == "extract_entities":
    SYSTEM_PROMPT = """
Ты — система извлечения именованных сущностей (NER). Извлеки из текста все организации, людей, даты и места.

Пример (Few-shot):
Текст: "Иван Петров из компании Рога и Копыта вчера посетил Москву."
Результат: {"persons": ["Иван Петров"], "organizations": ["Рога и Копыта"], "locations": ["Москва"], "dates": ["вчера"]}

Теперь обработай следующий текст и верни результат строго в JSON.
"""
    USER_TEMPLATE = "Текст:\n\n{text}"

elif TASK == "classify":
    SYSTEM_PROMPT = """
Ты — классификатор тональности текста. Определи, позитивный, негативный или нейтральный текст.

Примеры (Few-shot):
Текст: "Отличный фильм, всем рекомендую!"
Тональность: "позитивная"

Текст: "Ужасное обслуживание, больше не приду."
Тональность: "негативная"

Теперь определи тональность следующего текста и верни JSON:
{
  "sentiment": "позитивная/негативная/нейтральная",
  "confidence": 0.95
}
"""
    USER_TEMPLATE = "Текст для анализа:\n\n{text}"

else:
    print("❌ Неизвестная задача. Доступны: summarize, extract_entities, classify")
    exit(1)

# Обработка текстов
results = []
total_tokens = 0

for idx, text in enumerate(texts):
    user_prompt = USER_TEMPLATE.format(text=text)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P
        )
        content = response.choices[0].message.content
        tokens = response.usage.total_tokens
        total_tokens += tokens

        # Попытка распарсить JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"raw": content}

        # Сохраняем результат в зависимости от задачи
        if TASK == "summarize":
            summary = data.get("summary", content)
            keywords = ", ".join(data.get("keywords", []))
            results.append({
                "id": idx,
                "original": text,
                "summary": summary,
                "keywords": keywords,
                "tokens": tokens
            })
        elif TASK == "extract_entities":
            persons = ", ".join(data.get("persons", []))
            orgs = ", ".join(data.get("organizations", []))
            locs = ", ".join(data.get("locations", []))
            dates = ", ".join(data.get("dates", []))
            results.append({
                "id": idx,
                "original": text,
                "persons": persons,
                "organizations": orgs,
                "locations": locs,
                "dates": dates,
                "raw": content,
                "tokens": tokens
            })
        elif TASK == "classify":
            sentiment = data.get("sentiment", "неизвестно")
            confidence = data.get("confidence", 0.0)
            results.append({
                "id": idx,
                "original": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "tokens": tokens
            })

        print(f"✅ Обработан {idx+1}/{len(texts)} (токенов: {tokens})")

    except Exception as e:
        print(f"❌ Ошибка при обработке текста {idx}: {e}")
        results.append({
            "id": idx,
            "original": text,
            "error": str(e),
            "tokens": 0
        })

# Сохранение результатов в CSV
fieldnames = results[0].keys() if results else ["id", "original", "error"]
with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n🎉 Готово! Результаты сохранены в {OUTPUT_FILE}")
print(f"📊 Всего использовано токенов: {total_tokens}")