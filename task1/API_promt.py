import csv
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ================== НАСТРОЙКИ ==================
# Адрес твоего локального сервера (Ollama/vLLM)
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "not-needed"

# Имя модели.
MODEL = "qwen/qwen3.6-27b"

# Параметры для строгого вывода JSON
TEMPERATURE = 0.1
TOP_P = 0.5

# Тип задачи: "summarize", "extract_entities" или "classify"
TASK = "summarize"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
# ==============================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def get_system_prompt(task: str) -> str:
    """Формирует жесткий системный промпт с примерами и инструкциями."""
    if task == "summarize":
        return """
Ты — строгий JSON-генератор. Твоя задача — вернуть ТОЛЬКО валидный JSON-объект.
НИКАКИХ пояснений, никаких слов до или после JSON, никаких Markdown-блоков (```).
Формат ответа:
{
  "summary": "одно предложение с сутью текста",
  "keywords": ["ключевое слово 1", "ключевое слово 2"]
}
Цепочка рассуждений (выполняется внутри тебя, не выводится в ответ):
1. Выпиши 3 главных факта из текста.
2. Сформулируй резюме на их основе (1-2 предложения).
3. Сформируй список из 3-5 ключевых слов.
4. Оберни результат в JSON строго по шаблону выше.
ВАЖНО: Если ты добавишь хоть один символ вне фигурных скобок {}, задача считается проваленной.
"""
    elif task == "extract_entities":
        return """
Ты — система распознавания именованных сущностей (NER). Верни ТОЛЬКО корректный JSON-объект.
Никаких пояснений, никакого текста до или после JSON.
Формат JSON:
{
  "persons": ["Иван Петров"],
  "organizations": ["Рога и копыта"],
  "locations": ["Москва"],
  "dates": ["вчера"]
}
Извлеки сущности строго по этим категориям. Не выдумывай несуществующие сущности.
"""
    elif task == "classify":
        return """
Ты — классификатор тональности. Верни ТОЛЬКО корректный JSON-объект.
Никаких пояснений, никакого текста до или после JSON.
Формат JSON:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}
Confidence должен быть числом от 0.0 до 1.0.
"""
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def read_texts_from_csv(path: str) -> List[str]:
    """Читает CSV, пробуя разные кодировки для совместимости."""
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
) -> Tuple[Optional[str], int]:
    """
    Вызывает модель с повторными попытками.
    Возвращает кортеж: (текст ответа, количество токенов).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                response_format={"type": "json_object"}
            )
            content = response.choices.message.content
            tokens = response.usage.total_tokens
            return content, tokens
        except Exception as e:
            last_exc = e
            print(f"Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * attempt
                print(f"   Ожидание {delay} сек перед следующей попыткой...")
                time.sleep(delay)
    print(f"Все попытки исчерпаны. Последняя ошибка: {last_exc}")
    return None, 0

def parse_json_safe(content: Optional[str]) -> Dict[str, Any]:
    """
    Пытается распарсить JSON. Если не получается, ищет JSON внутри текста.
    Всегда возвращает словарь с полями 'data' (или 'error') и 'raw'.
    """
    if not content:
        return {"error": "Нет ответа от модели", "raw": "", "data": None}
    raw_response = content

    # Попытка 1: Прямой парсинг (если модель вернула чистый JSON благодаря response_format)
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return {"data": data, "raw": raw_response, "error": None}
    except json.JSONDecodeError:
        pass

    # Попытка 2: Поиск JSON через Regex (если модель добавила преамбулу несмотря на инструкции)
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return {"data": data, "raw": raw_response, "error": None}
        except Exception:
            pass

    # Попытка 3: Поиск внутри Markdown блока ```json ... ```
    code_block_match = re.search(r"```json\s*([\s\S]*?)```", content)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return {"data": data, "raw": raw_response, "error": None}
        except Exception:
            pass

    # Если ничего не помогло
    return {"error": "Не удалось распарсить JSON", "raw": raw_response, "data": None}
def main():
    # Чтение входных данных
    try:
        texts = read_texts_from_csv(INPUT_FILE)
    except RuntimeError as e:
        print(f"Ошибка чтения файла: {e}")
        return
    if not texts:
        print("Нет текстов для обработки.")
        # Создаем пустой CSV с заголовками
        base_fields = ["id", "original"]
        if TASK == "summarize":
            fieldnames = base_fields + ["summary", "keywords", "tokens", "error", "raw_response"]
        elif TASK == "extract_entities":
            fieldnames = base_fields + ["persons", "organizations", "locations", "dates", "tokens", "error", "raw_response"]
        elif TASK == "classify":
            fieldnames = base_fields + ["sentiment", "confidence", "tokens", "error", "raw_response"]
        else:
            fieldnames = base_fields + ["tokens", "error", "raw_response"]

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

        # Получение ответа и токенов
        content, tokens_used = call_llm_with_retry(
            messages=messages,
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        total_tokens += tokens_used

        # Парсинг ответа
        parse_result = parse_json_safe(content)
        data = parse_result["data"]
        error_msg = parse_result["error"]
        raw_response = parse_result["raw"]

        # Вывод сырого ответа в консоль для отладки (если есть ошибка)
        if error_msg:
            print(f"   [ОТЛАДКА] Сырой ответ модели (проблема с форматом):")
            print(raw_response[:500]) # Вывод первых 500 символов
            print("   [ОТЛАДКА] Конец сырого ответа")
        if error_msg:
            results.append({
                "id": idx,
                "original": text,
                "summary": "",
                "keywords": "",
                "tokens": tokens_used,
                "error": error_msg,
                "raw_response": raw_response
            })
            print(f"   Ошибка: {error_msg}")
            continue

        # Формирование результата в зависимости от задачи
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
                "raw_response": raw_response
            })
        elif TASK == "extract_entities":
            persons = data.get("persons", [])
            orgs = data.get("organizations", [])
            locs = data.get("locations", [])
            dates = data.get("dates", [])

            # Защита от не-списков
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
                "tokens": tokens_used,
                "error": "",
                "raw_response": raw_response
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
                "raw_response": raw_response
            })
        print(f"   Готово (токены: {tokens_used})")

    # Определение заголовков CSV
    base_fields = ["id", "original"]
    if TASK == "summarize":
        fieldnames = base_fields + ["summary", "keywords", "tokens", "error", "raw_response"]
    elif TASK == "extract_entities":
        fieldnames = base_fields + ["persons", "organizations", "locations", "dates", "tokens", "error", "raw_response"]
    elif TASK == "classify":
        fieldnames = base_fields + ["sentiment", "confidence", "tokens", "error", "raw_response"]
    else:
        fieldnames = base_fields + ["tokens", "error", "raw_response"]

    # Запись результатов
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nГотово. Результаты сохранены в {OUTPUT_FILE}")
    print(f"Всего токенов использовано: {total_tokens}")

if __name__ == "__main__":
    main()