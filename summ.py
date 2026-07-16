import csv
import json
from openai import OpenAI

# --- Настройка клиента ---
client = OpenAI(
    base_url="http://192.168.8.11:1234/v1",
    api_key="not-needed"
)

# --- Модель ---
MODEL = "qwen/qwen3.6-27b"

# --- Вход / выход ---
input_file = "input.csv"
output_file = "output.csv"

# --- Автоматическое определение кодировки ---
encodings = ["utf-8-sig", "cp1251", "utf-8"]
texts = []
for enc in encodings:
    try:
        with open(input_file, "r", encoding=enc) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text") and row["text"].strip():
                    texts.append(row["text"])
        print(f"File read with encoding {enc}")
        break
    except (UnicodeDecodeError, KeyError):
        continue
else:
    print("Failed to read file with any encoding.")
    exit(1)

# --- Системный промпт с Few-shot и Chain-of-Thought ---
SYSTEM_PROMPT = """
You are an assistant that briefly summarizes texts.

Here are examples of how to respond (Few-shot):

Example 1:
Text: "Yesterday, OpenAI introduced a new GPT-5 model that outperforms previous versions across all metrics. The model is now available via API."
Key facts: OpenAI, new GPT-5 model, outperforms previous, available via API.
Summary: "OpenAI announced GPT-5 — a more powerful model, now available via API."

Example 2:
Text: "Researchers from Stanford found that regular walking improves cognitive function in the elderly by 20%."
Key facts: Stanford, walking, cognitive improvement by 20%, elderly.
Summary: "Walking improves memory in the elderly by 20% — Stanford study."

Now your task:
1. First extract key facts from the text (Chain-of-Thought).
2. Then give a brief summary (1-2 sentences).
3. Return the result in JSON format:
{
  "summary": "brief summary",
  "keywords": ["key1", "key2", "key3"]
}
"""

# --- Обработка каждого текста ---
results = []
total_tokens = 0

for idx, text in enumerate(texts):
    print(f"Processing {idx+1}/{len(texts)}...")
    try:
        user_prompt = f"Text to analyze:\n\n{text}"
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            top_p=0.9,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        total_tokens += tokens_used

        try:
            data = json.loads(content)
            summary = data.get("summary", "No summary")
            keywords = data.get("keywords", [])
        except json.JSONDecodeError:
            summary = content
            keywords = []

        results.append({
            "id": idx,
            "original": text,
            "summary": summary,
            "keywords": ", ".join(keywords),
            "tokens": tokens_used
        })
        print(f"   Done (tokens: {tokens_used})")

    except Exception as e:
        print(f"   Error: {e}")
        results.append({
            "id": idx,
            "original": text,
            "summary": "ERROR",
            "keywords": "",
            "tokens": 0
        })

# --- Сохраняем результаты в CSV ---
with open(output_file, "w", encoding="utf-8", newline="") as f:
    fieldnames = ["id", "original", "summary", "keywords", "tokens"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nDone. Results saved to {output_file}")
print(f"Total tokens used: {total_tokens}")