import csv
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
import httpx  # для гибкой настройки таймаутов

# ==================== НАСТРОЙКИ ====================

BASE_URL = "http://192.168.0.140:1234/v1"   # Адрес локального сервера
API_KEY = "not-needed"                      # Ключ не требуется для локальных серверов
MODEL = "qwen/qwen3.6-27b"                  # Имя модели

TEMPERATURE = 0.1   # Чем ниже, тем более детерминирован ответ
TOP_P = 0.5         # Выбор следующего токена только из наиболее вероятных вариантов

TASK = "summarize"  # Варианты: "summarize", "extract_entities", "classify"
INPUT_FILE = "input.csv"    # Входной CSV с колонкой "text"
OUTPUT_FILE = "output.csv"  # Результирующий CSV

MAX_RETRIES = 5             # Попытки повтора запроса при ошибке
RETRY_DELAY_SECONDS = 3     # Время задержки между попытками
TIMEOUT_SECONDS = 300       # Базовый таймаут на один запрос
# ===================================================

# Клиент OpenAI
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    timeout=httpx.Timeout(TIMEOUT_SECONDS, connect=10.0)
)

def get_system_prompt(task: str) -> str:
    """
    Возвращает системный промпт для выбранной задачи.
    Промпт жёстко требует от модели возвращать только JSON,
    чтобы упростить парсинг.
    """
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
    """
    Читает CSV-файл, извлекает тексты из колонки 'text'.
    Пробует несколько кодировок, чтобы избежать ошибок с кириллицей.
    """
    encodings = ["utf-8-sig", "cp1251", "utf-8"]  # распространённые кодировки
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                # Проверка нужной колонки
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
        max_tokens: int,
) -> Tuple[Optional[str], int]:
    """
    Отправляет запрос к модели с повторными попытками при ошибках.
    При каждой следующей попытке таймаут увеличивается,
    чтобы дать серверу больше времени, если он перегружен.
    Возвращает или при провале.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        current_timeout = TIMEOUT_SECONDS * attempt  # динамический таймаут
        try:
            print(f"   Попытка {attempt}/{MAX_RETRIES} (таймаут {current_timeout}с)...")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,      # ограничение длины ответа, чтобы ускорить генерацию
                timeout=current_timeout,    # передача увеличенного таймаута
            )
            content = response.choices[0].message.content
            tokens = response.usage.total_tokens
            return content, tokens
        except Exception as e:
            last_exc = e
            print(f"   Попытка {attempt}/{MAX_RETRIES} не удалась: {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_SECONDS * attempt
                print(f"   Ожидание {delay} сек перед следующей попыткой...")
                time.sleep(delay)
    print(f"Все попытки исчерпаны. Последняя ошибка: {last_exc}")
    return None, 0

def parse_json_safe(content: Optional[str]) -> Dict[str, Any]:
    """
    Пытается извлечь валидный JSON из ответа модели.
    Сначала пробует прямой парсинг, затем ищет JSON-блок с помощью регулярного выражения,
    а также проверяет наличие Markdown-блока ```json ... ```.
    Возвращает словарь с ключами: 'data' или 'error',
    а также 'raw'.
    """
    if not content:
        return {"error": "Нет ответа от модели", "raw": "", "data": None}
    raw_response = content

    # Попытка 1: прямой парсинг всего ответа
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return {"data": data, "raw": raw_response, "error": None}
    except json.JSONDecodeError:
        pass

    # Попытка 2: поиск первой фигурной скобки и всего до последней закрывающей
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return {"data": data, "raw": raw_response, "error": None}
        except Exception:
            pass

    # Попытка 3: извлечение из Markdown-блока ```json ... ```
    code_block_match = re.search(r"```json\s*([\s\S]*?)```", content)
    if code_block_match:
        json_str = code_block_match.group(1).strip()
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                return {"data": data, "raw": raw_response, "error": None}
        except Exception:
            pass

    return {"error": "Не удалось распарсить JSON", "raw": raw_response, "data": None}

def get_max_tokens_for_task(task: str) -> int:
    """
    Возвращает разумное ограничение на количество генерируемых токенов
    в зависимости от задачи. Это помогает ускорить ответ, так как модель
    не тратит время на генерацию лишнего текста.
    """
    if task == "summarize":
        return 150      # краткое резюме
    elif task == "extract_entities":
        return 200      # несколько сущностей
    elif task == "classify":
        return 50       # всего два поля
    else:
        return 200

def main():
    """
    Основная функция:
      - читает входной CSV,
      - для каждого текста формирует запрос,
      - вызывает модель с повторными попытками,
      - парсит JSON-ответ,
      - сохраняет все результаты в выходной CSV.
    """
    start_time = time.time()

    # Шаг 1. Чтение текстов
    try:
        texts = read_texts_from_csv(INPUT_FILE)
    except RuntimeError as e:
        print(f"Ошибка чтения файла: {e}")
        return

    # Если текстов нет – создаём пустой выходной файл с правильными заголовками
    if not texts:
        print("Нет текстов для обработки.")
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

    # Шаг 2. Подготовка промптов и ограничений
    system_prompt = get_system_prompt(TASK)
    max_tokens = get_max_tokens_for_task(TASK)
    results: List[Dict[str, Any]] = []
    total_tokens = 0

    # Шаг 3. Обработка каждого текста
    for idx, text in enumerate(texts):
        print(f"\nОбработка {idx+1}/{len(texts)}... (длина текста: {len(text)} симв.)")
        user_prompt = f"Текст для анализа:\n\n{text}"
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Замеряем время выполнения запроса
        req_start = time.time()
        content, tokens_used = call_llm_with_retry(
            messages=messages,
            model=MODEL,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=max_tokens,
        )
        req_time = time.time() - req_start
        total_tokens += tokens_used

        # Парсим ответ
        parse_result = parse_json_safe(content)
        data = parse_result["data"]
        error_msg = parse_result["error"]
        raw_response = parse_result["raw"]

        # Если при парсинге возникла ошибка – выводим сырой ответ для отладки
        if error_msg:
            print(f"[ОТЛАДКА] Сырой ответ модели (первые 500 символов):")
            print(raw_response[:500])
            print("[ОТЛАДКА] Конец сырого ответа")

        # Формируем строку результата
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
            print(f"   Ошибка: {error_msg} (время запроса: {req_time:.2f}с)")
            continue

        # В зависимости от задачи заполняем соответствующие поля
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
            # Приводим все списки к типу list (на случай, если модель вернёт строку или None)
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
        print(f"   Готово (токены: {tokens_used}, время: {req_time:.2f}с)")

    # Шаг 4. Определяем заголовки выходного CSV в зависимости от задачи
    base_fields = ["id", "original"]
    if TASK == "summarize":
        fieldnames = base_fields + ["summary", "keywords", "tokens", "error", "raw_response"]
    elif TASK == "extract_entities":
        fieldnames = base_fields + ["persons", "organizations", "locations", "dates", "tokens", "error", "raw_response"]
    elif TASK == "classify":
        fieldnames = base_fields + ["sentiment", "confidence", "tokens", "error", "raw_response"]
    else:
        fieldnames = base_fields + ["tokens", "error", "raw_response"]

    # Шаг 5. Запись результатов в CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    total_time = time.time() - start_time
    print(f"\nГотово. Результаты сохранены в {OUTPUT_FILE}")
    print(f"Всего токенов использовано: {total_tokens}")
    print(f"Общее время выполнения: {total_time:.2f} секунд")
if __name__ == "__main__":
    main()