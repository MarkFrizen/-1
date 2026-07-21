import csv
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ================= НАСТРОЙКИ =================
# Адрес локального сервера LM Studio (должен быть доступен по сети)
BASE_URL = "http://192.168.8.11:1234/v1"
# Для локального сервера ключ не требуется, но клиент OpenAI ожидает какое-то значение
API_KEY = "not-needed"
# ID модели должен точно совпадать с тем, что возвращает эндпоинт /v1/models
MODEL = "qwen/qwen3.6-35b-a3b"

# Тип задачи для обработки текстов: "summarize", "extract_entities" или "classify"
TASK = "summarize"
# Входной CSV-файл (должен содержать колонку text)
INPUT_FILE = "input.csv"
# Выходной CSV-файл с результатами
OUTPUT_FILE = "output.csv"

# Параметры генерации для контроля «креативности» и разнообразия ответов
TEMPERATURE = 0.1   # Низкая температура — более предсказуемые, стабильные ответы
TOP_P = 0.5          # Top-p сэмплирование ограничивает выборку токенов по вероятности
MAX_TOKENS = 150    # Лимит токенов на ответ — помогает ускорить генерацию и снизить таймауты

# Настройки повторных попыток при сетевых ошибках
MAX_RETRIES = 2      # Количество ретраев (не для таймаутов, а для разрывов соединения)
RETRY_DELAY = 5      # Задержка между попытками в секундах

# Таймаут в секундах (30 минут). Критически важен для больших моделей (35B) на слабых GPU:
# генерация может занимать несколько минут из-за низкой скорости токенов/сек.
TIMEOUT_SECONDS = 1800
# =============================================

# Инициализация клиента OpenAI с увеличенным таймаутом
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT_SECONDS)

def get_prompt(task: str) -> str:
    """
    Формирует системный промпт в зависимости от типа задачи.
    Промпты сделаны максимально лаконичными (без избыточного Chain-of-Thought),
    чтобы уменьшить время обработки промпта (TTFT) на моделях с малым VRAM.
    В каждом варианте:
      - чёткое требование вернуть только валидный JSON;
      - формат JSON;
      - один пример (Few-shot);
      - указание «Текст для анализа:» в конце, чтобы модель понимала, где начинаются данные.
    """
    if task == "summarize":
        return """
You are a summary generator. Return ONLY a valid JSON object. No explanations, no text outside braces.

Format:
{
  "summary": "one sentence with the essence of the text",
  "keywords": ["keyword 1", "keyword 2"]
}

Example (follow this format strictly):
Text: "The company opened a new plant in Siberia. Production will start in October."
Answer: {"summary": "The company is opening a new plant in Siberia with launch in October.", "keywords": ["plant", "Siberia", "production"]}

Text for analysis:
"""
    elif task == "extract_entities":
        return """
You are an entity extraction system. Return ONLY a correct JSON object. No explanations.

Format:
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "dates": []
}

Example:
Text: "Ivan Petrov from the company Horns and Hooves arrived from Moscow yesterday."
Answer: {"persons": ["Ivan Petrov"], "organizations": ["Horns and Hooves"], "locations": ["Moscow"], "dates": ["yesterday"]}

Text for analysis:
"""
    elif task == "classify":
        return """
You are a sentiment classifier. Return ONLY JSON. No explanations.

Format:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}

Example:
Text: "Excellent service, everything is fast and convenient."
Answer: {"sentiment": "positive", "confidence": 0.95}

Text for analysis:
"""
    else:
        raise ValueError(f"Unknown task: {task}")

def read_csv(path: str) -> List[str]:
    """
    Читает CSV-файл, пытаясь несколько кодировок (utf-8-sig, cp1251, utf-8).
    Возвращает список строк из колонки 'text', пропуская пустые значения.
    Если файл не найден или колонка отсутствует — выбрасывает RuntimeError.
    """
    encodings = ["utf-8-sig", "cp1251", "utf-8"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    continue
                texts = [row["text"].strip() for row in reader if row.get("text", "").strip()]
                print(f"Read texts: {len(texts)} (encoding {enc})")
                return texts
        except Exception:
            continue
    raise RuntimeError("Failed to read CSV with any encoding.")

def call_with_retry(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    """
    Отправляет запрос к модели с возможностью повторной попытки при сетевых ошибках.

    Логика:
      - Ретраи делаются только при ConnectionError (разрыв соединения).
      - Таймауты (Timeout) не ретраятся — вместо этого используется большой глобальный timeout.
      - При других ошибках (HTTPError и т.п.) функция сразу возвращает None.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
            )
            return resp.choices[0].message.content
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e)
            # Проверяем, является ли ошибка сетевой (ConnectionError или упоминание connection)
            if "ConnectionError" in err_name or "connection" in err_str.lower():
                print(f"[Attempt {attempt}/{MAX_RETRIES}] Connection error: {e}")
                if attempt < MAX_RETRIES:
                    print(f"Waiting {RETRY_DELAY} seconds before next attempt...")
                    time.sleep(RETRY_DELAY)
                continue
            else:
                # Другие ошибки (в т.ч. таймаут) не повторяем — считаем их критическими
                print(f"[Attempt {attempt}/{MAX_RETRIES}] Critical error: {e}")
                return None
    return None

def parse_json_strict(content: Optional[str]) -> Dict[str, Any]:
    """
    Строго парсит ответ модели в JSON.
    Особенности:
      - Удаляет возможные Markdown-обёртки (```json ... ```), если модель их добавляет.
      - Возвращает словарь с полями:
          * data — распарсенные данные (dict) или None;
          * error — описание ошибки или None;
          * raw — исходный ответ модели.
    """
    result = {"data": None, "error": None, "raw": content or ""}
    if not content:
        result["error"] = "No response from model"
        return result

    # Удаляем Markdown-блоки, если они есть
    clean_content = content.strip()
    if clean_content.startswith("```json"):
        clean_content = clean_content[7:]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    clean_content = clean_content.strip()
    try:
        data = json.loads(clean_content)
        if isinstance(data, dict):
            result["data"] = data
        else:
            result["error"] = "JSON is not an object (dict)"
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {e}"
    return result


def main():
    # Читаем тексты из CSV
    texts = read_csv(INPUT_FILE)
    if not texts:
        print("No texts to process.")
        return

    # Получаем нужный системный промпт
    system_prompt = get_prompt(TASK)
    results = []
    total_tokens = 0

    # Обрабатываем каждый текст
    for i, text in enumerate(texts, start=1):
        print(f"\n--- Processing {i}/{len(texts)} ---")

        # Формируем сообщения: system (промпт) + user (сам текст)
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        # Вызываем модель с ретраями
        content = call_with_retry(messages)
        parsed = parse_json_strict(content)
        data = parsed["data"]
        error = parsed["error"]
        raw = parsed["raw"]

        # Грубая оценка токенов (для статистики): длина текста / 4 + лимит ответа
        tokens_used = len(text) // 4 + MAX_TOKENS
        total_tokens += tokens_used
        row = {
            "id": i,
            "original": text,
            "tokens": tokens_used,
            "error": error or "",
            "raw_response": raw,
        }

        # Если произошла ошибка, заполняем пустые значения в зависимости от задачи
        if error:
            if TASK == "summarize":
                row["summary"] = ""
                row["keywords"] = ""
            elif TASK == "extract_entities":
                for k in ["persons", "organizations", "locations", "dates"]:
                    row[k] = ""
            elif TASK == "classify":
                row["sentiment"] = ""
                row["confidence"] = 0.0
            results.append(row)
            print(f"Error: {error}")
            continue

        # Заполняем поля результата в зависимости от выбранной задачи
        if TASK == "summarize":
            row["summary"] = data.get("summary", "")
            kw = data.get("keywords", [])
            row["keywords"] = ", ".join(kw) if isinstance(kw, list) else ""
        elif TASK == "extract_entities":
            for k in ["persons", "organizations", "locations", "dates"]:
                lst = data.get(k, [])
                row[k] = ", ".join(lst) if isinstance(lst, list) else ""
        elif TASK == "classify":
            row["sentiment"] = data.get("sentiment", "unknown")
            conf = data.get("confidence", 0.0)
            row["confidence"] = float(conf) if isinstance(conf, (int, float)) else 0.0
        results.append(row)
        print(f"Successfully processed: {i}")

    # Определяем нужные колонки для выходного CSV в зависимости от задачи
    base_cols = ["id", "original", "tokens", "error", "raw_response"]
    if TASK == "summarize":
        fieldnames = base_cols + ["summary", "keywords"]
    elif TASK == "extract_entities":
        fieldnames = base_cols + ["persons", "organizations", "locations", "dates"]
    elif TASK == "classify":
        fieldnames = base_cols + ["sentiment", "confidence"]
    else:
        fieldnames = base_cols

    # Записываем результаты в CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDone. Results in {OUTPUT_FILE}")
    print(f"Processed texts: {len(results)}, estimated tokens: {total_tokens}")
if __name__ == "__main__":
    main()