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
You are an assistant that briefly summarizes texts. Return ONLY a valid JSON object, no text before or after.

Examples (Few-shot):
Text: "Yesterday, company X announced record profits."
Key facts: company X, record profits.
Summary: "Company X reported record profits."
JSON:
{
  "summary": "Company X reported record profits.",
  "keywords": ["company X", "record profits"]
}

Text: "Scientists discovered a new dinosaur species in Argentina."
Key facts: scientists, new dinosaur species, Argentina.
Summary: "A new dinosaur species found in Argentina."
JSON:
{
  "summary": "A new dinosaur species found in Argentina.",
  "keywords": ["scientists", "dinosaur species", "Argentina"]
}

Now:
1. First extract key facts (Chain-of-Thought).
2. Then give a brief summary (1-2 sentences).
3. Return ONLY JSON:
{
  "summary": "brief summary",
  "keywords": ["key1", "key2", "key3"]
}
NO EXPLANATIONS, NO TEXT BEFORE/AFTER JSON.
"""
    elif task == "extract_entities":
        return """
You are a Named Entity Recognition (NER) system. Return ONLY a valid JSON object, no text before or after.

Example:
Text: "Ivan Petrov from Horns and Hooves company visited Moscow yesterday."
Result:
{
  "persons": ["Ivan Petrov"],
  "organizations": ["Horns and Hooves"],
  "locations": ["Moscow"],
  "dates": ["yesterday"]
}

Return JSON with keys: persons, organizations, locations, dates. NO EXPLANATIONS.
"""
    elif task == "classify":
        return """
You are a sentiment classifier. Return ONLY a valid JSON object, no text before or after.

Examples:
Text: "Great movie, I recommend it to everyone!"
Result: {"sentiment": "positive", "confidence": 0.95}

Text: "Terrible service, I will never come back."
Result: {"sentiment": "negative", "confidence": 0.98}

Return JSON:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}
NO EXPLANATIONS.
"""
    else:
        raise ValueError(f"Unknown task: {task}")


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
                print(f"File read with encoding {enc}, texts count: {len(texts)}")
                return texts
        except (UnicodeDecodeError, KeyError, FileNotFoundError):
            continue
    raise RuntimeError("Failed to read file with any encoding.")


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
                # response_format убран, чтобы избежать 400 на локальных серверах
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return content
        except Exception as e:
            last_exc = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)
    print(f"All retries failed. Last error: {last_exc}")
    return None


def parse_json_safe(content: Optional[str]) -> Dict[str, Any]:
    if not content:
        return {"error": "No response from model"}
    try:
        data = json.loads(content)
        if not isinstance(data, dict):
            return {"error": "Response is not a JSON object", "raw": content}
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
        return {"error": f"JSON decode error: {e}", "raw": content}


def main():
    texts = read_texts_from_csv(INPUT_FILE)
    if not texts:
        print("No texts to process.")
        fieldnames = ["id", "original", "summary", "keywords", "tokens", "error"]
        with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    system_prompt = get_system_prompt(TASK)
    results: List[Dict[str, Any]] = []
    total_tokens = 0

    for idx, text in enumerate(texts):
        print(f"Processing {idx+1}/{len(texts)}...")
        user_prompt = f"Text to analyze:\n\n{text}"

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
        # Если есть response, можно взять tokens, но при ошибках response может не быть
        # Для простоты считаем tokens только при успешном ответе
        if content is not None:
            # tokens мы не можем получить без объекта response, поэтому пока оставим 0
            # либо можно модифицировать call_llm_with_retry, чтобы он возвращал (content, tokens)
            pass

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
            print(f"   Error: {error_msg}")
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
                "error": "Unknown task",
            })

        print(f"   Done (tokens: {tokens_used})")

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

    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    print(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    main()
