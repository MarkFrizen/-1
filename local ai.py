import csv
import json
from openai import OpenAI

# ================== НАСТРОЙКИ ==================
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "not-needed"
MODEL = "qwen/qwen3.6-27b"
TEMPERATURE = 0.5
TOP_P = 0.9
TASK = "summarize"                  # варианты: summarize, extract_entities, classify
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
        print(f"File read with encoding {enc}")
        break
    except (UnicodeDecodeError, KeyError):
        continue
else:
    print("Failed to read file.")
    exit(1)

# Промпты в зависимости от задачи
if TASK == "summarize":
    SYSTEM_PROMPT = """
You are an assistant that briefly summarizes texts.

Examples (Few-shot):
Text: "Yesterday, company X announced record profits."
Summary: "Company X reported record profits."

Text: "Scientists discovered a new dinosaur species in Argentina."
Summary: "A new dinosaur species found in Argentina."

Now summarize the following text. First extract key facts (Chain-of-Thought), then give a brief summary in JSON:
{
  "summary": "brief summary",
  "keywords": ["key1", "key2"]
}
"""
    USER_TEMPLATE = "Text to summarize:\n\n{text}"

elif TASK == "extract_entities":
    SYSTEM_PROMPT = """
You are a Named Entity Recognition (NER) system. Extract all organizations, persons, dates, and locations from the text.

Example (Few-shot):
Text: "Ivan Petrov from Horns and Hooves company visited Moscow yesterday."
Result: {"persons": ["Ivan Petrov"], "organizations": ["Horns and Hooves"], "locations": ["Moscow"], "dates": ["yesterday"]}

Now process the following text and return JSON.
"""
    USER_TEMPLATE = "Text:\n\n{text}"

elif TASK == "classify":
    SYSTEM_PROMPT = """
You are a sentiment classifier. Determine if the text is positive, negative, or neutral.

Examples (Few-shot):
Text: "Great movie, I recommend it to everyone!"
Sentiment: "positive"

Text: "Terrible service, I will never come back."
Sentiment: "negative"

Now classify the following text and return JSON:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.95
}
"""
    USER_TEMPLATE = "Text to analyze:\n\n{text}"

else:
    print("Unknown task. Available: summarize, extract_entities, classify")
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

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"raw": content}

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
            sentiment = data.get("sentiment", "unknown")
            confidence = data.get("confidence", 0.0)
            results.append({
                "id": idx,
                "original": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "tokens": tokens
            })

        print(f"Processed {idx+1}/{len(texts)} (tokens: {tokens})")

    except Exception as e:
        print(f"Error processing text {idx}: {e}")
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

print(f"\nDone. Results saved to {OUTPUT_FILE}")
print(f"Total tokens used: {total_tokens}")