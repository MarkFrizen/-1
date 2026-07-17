import csv
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ================== НАСТРОЙКИ ==================
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "not-needed"
MODEL = "qwen/qwen3.6-27b"

TEMPERATURE = 0.5
TOP_P = 0.9
TASK = "summarize"  # summarize, extract_entities, classify

INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
# ==============================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def get_system_prompt(task: str) -> str:
    if task == "summarize":
        return """
Ты — ассистент, который кратко суммирует тексты. Верни ТОЛЬКО корректный JSON-объект, без какого-либо текста до или после него.

Примеры (few-shot):
Текст: «Вчера компания X объявила о рекордной прибыли».
Ключевые факты: компания X, рекордная прибыль.
Резюме: «Компания X сообщила о рекордной прибыли».
JSON:
{
  "summary": "Компания X сообщила о рекордной прибыли.",
  "keywords": ["компания X", "рекордная прибыль"]
}

Текст: «Учёные обнаружили новый вид динозавра в Аргентине».
Ключевые факты: учёные, новый вид динозавра, Аргентина.
Резюме: «В Аргентине найден новый вид динозавра».
JSON:
{
  "summary": "В Аргентине найден новый вид динозавра.",
  "keywords": ["учёные", "вид динозавра", "Аргентина"]
}

Теперь:
1. Сначала извлеки ключевые факты (цепочка рассуждений).
2. Затем дай краткое резюме (1–2 предложения).
3. Верни ТОЛЬКО JSON:
{
  "summary": "краткое резюме",
  "keywords": ["ключ1", "ключ2", "ключ3"]
}
НИКАКИХ ПОЯСНЕНИЙ, НИКАКОГО ТЕКСТА ДО/ПОСЛЕ JSON.
"""
    elif task == "extract_entities":
        return """
Ты — система распознавания именованных сущностей (NER). Верни ТОЛЬКО корректный JSON-объект, без какого-либо текста до или после него.

Пример:
Текст: «Иван Петров из компании „Рога и копыта“ посетил Москву вчера».
Результат:
{
  "persons": ["Иван Петров"],
  "organizations": ["Рога и копыта"],
  "locations": ["Москва"],
  "dates": ["вчера"]
}

Верни JSON с ключами: persons, organizations, locations, dates. НИКАКИХ ПОЯСНЕНИЙ.
"""
    elif task == "classify":
        return """
Ты — классификатор тональности. Верни ТОЛЬКО корректный JSON-объект, без какого-либо текста до или после него.

Примеры:
Текст: «Отличный фильм, рекомендую всем!»
Результат: {"sentiment": "positive", "confidence": 0.95}

Текст: «Ужасный сервис, больше никогда не приду».
Результат: {"sentiment": "negative", "confidence": 0.98}

Верни JSON:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}
НИКАКИХ ПОЯСНЕНИЙ.
"""
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def read_texts_from_csv(path: str) -> List[str]:
    encodings = ["utf-8-sig", "cp1251", "utf-8"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    continue
                texts = []
                for row in reader:
                    t = row.get("text", "")
                    if t and t.strip():
                        texts.append(t.strip())
                print(f"Файл прочитан с кодировкой {enc}, количество текстов: {len(texts)}")
                return texts
        except (UnicodeDecodeError, KeyError, FileNotFoundError):
            continue
    raise RuntimeError("Не удалось прочитать файл ни с одной из кодировок.")

def call_llm_with_retry(
        messages: List[ChatCompletionMessageParam],
        model: str,
        temperature: float,
        top_p: float,
) -> Optional[str]:
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            content = response.choices.message.content
            return content
        except Exception as e:
            last_exc = e
            print(f"Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * attempt
                print(f"   Ожидание {delay} сек перед следующей попыткой...")
                time.sleep(delay)
    print(f"Все попытки исчерпаны. Последняя ошибка: {last_exc}")
    return None


def parse_json_safe(content: Optional[str]) -> Dict[str, Any]:
    if not content:
        return {"error": "Нет ответа от модели"}
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return {"error": "Ответ не является JSON-объектом", "raw": content}
        return data
    except json.JSONDecodeError as e:
        # Пытаемся найти JSON внутри текста (на случай, если модель добавила преамбулу)
        import re
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {"error": f"Ошибка декодирования JSON: {e}", "raw": content}


def main():
    texts = read_texts_from_csv(INPUT_FILE)
    if not texts:
        print("Нет текстов для обработки.")
        fieldnames = ["id", "original", "summary", "keywords", "tokens", "error"]
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    system_prompt = get_system_prompt(TASK)
    results: List[Dict[str, Any]] = []
    total_tokens = 0

    for idx, text in enumerate(texts):
        print(f"Обработка {idx+1}/{len(texts)}...")
        user_prompt = f"Текст для анализа:\n\n{text}"

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        content = call_llm_with_retry(
            messages=messages,
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

        tokens_used = 0
        # Токены не сохраняются, так как функция call_llm_with_retry возвращает только content.
        # Для учёта токенов нужно модифицировать функцию, чтобы она возвращала (content, tokens).

        data = parse_json_safe(content)

        error_msg = data.get("error")
        if error_msg:
            results.append({
                "id": idx,
                "original": text,
                "summary": "",
                "keywords": "",
                "tokens": tokens_used,
                "error": error_msg,
            })
            print(f"   Ошибка: {error_msg}")
            continue

        if TASK == "summarize":
            summary = data.get("summary", "")
            keywords = data.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            results.append({
                "id": idx,
                "original": text,
                "summary": summary,
                "keywords": ", ".join(keywords),
                "tokens": tokens_used,
                "error": "",
            })

        elif TASK == "extract_entities":
            persons = data.get("persons", [])
            orgs = data.get("organizations", [])
            locs = data.get("locations", [])
            dates = data.get("dates", [])
            for lst in [persons, orgs, locs, dates]:
                if not isinstance(lst, list):
                    lst = []
            results.append({
                "id": idx,
                "original": text,
                "persons": ", ".join(persons),
                "organizations": ", ".join(orgs),
                "locations": ", ".join(locs),
                "dates": ", ".join(dates),
                "raw": json.dumps(data),
                "tokens": tokens_used,
                "error": "",
            })

        elif TASK == "classify":
            sentiment = data.get("sentiment", "unknown")
            confidence = data.get("confidence", 0.0)
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            results.append({
                "id": idx,
                "original": text,
                "sentiment": sentiment,
                "confidence": confidence,
                "tokens": tokens_used,
                "error": "",
            })

        else:
            results.append({
                "id": idx,
                "original": text,
                "summary": "",
                "keywords": "",
                "tokens": tokens_used,
                "error": "Неизвестная задача",
            })

        print(f"   Готово (токены: {tokens_used})")

    base_fields = ["id", "original"]
    if TASK == "summarize":
        fieldnames = base_fields + ["summary", "keywords", "tokens", "error"]
    elif TASK == "extract_entities":
        fieldnames = base_fields + ["persons", "organizations", "locations", "dates", "raw", "tokens", "error"]
    elif TASK == "classify":
        fieldnames = base_fields + ["sentiment", "confidence", "tokens", "error"]
    else:
        fieldnames = base_fields + ["tokens", "error"]

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nГотово. Результаты сохранены в {OUTPUT_FILE}")
    print(f"Всего токенов использовано: {total_tokens}")


if __name__ == "__main__":
    main()