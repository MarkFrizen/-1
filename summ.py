import csv
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.140:1234/v1",
    api_key="not-needed"  # если аутентификация отключена
)

input_file = "input.csv"
output_file = "output.csv"

# Автоподбор кодировки
encodings = ["utf-8-sig", "cp1251", "utf-8"]
texts = []
for enc in encodings:
    try:
        with open(input_file, "r", encoding=enc) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("text") and row["text"].strip():
                    texts.append(row["text"])
        print(f"Файл прочитан в кодировке {enc}")
        break
    except (UnicodeDecodeError, KeyError):
        continue
else:
    print("Не удалось прочитать файл.")
    exit(1)

results = []
for idx, text in enumerate(texts):
    prompt = f"Кратко перескажи следующий текст:\n\n{text}"
    try:
        response = client.chat.completions.create(
            model="qwen/qwen3.6-35b-a3b",
            messages=[
                {"role": "system", "content": "Ты — помощник, который кратко пересказывает тексты."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            top_p=0.9
        )
        summary = response.choices[0].message.content
        results.append({"id": idx, "original": text, "summary": summary})
        print(f"Обработан {idx+1}/{len(texts)}")
    except Exception as e:
        print(f"Ошибка: {e}")
        results.append({"id": idx, "original": text, "summary": "ОШИБКА"})

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "original", "summary"])
    writer.writeheader()
    writer.writerows(results)

print(f"Готово! Результаты в {output_file}")