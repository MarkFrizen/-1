import csv
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ================= НАСТРОЙКИ =================
# Адрес локального сервера LM Studio (должен быть доступен по сети)
BASE_URL = "http://192.168.8.11:1234/v1"
# Для локального сервера ключ не требуется, но клиент OpenAI ожидает какое‑то значение
API_KEY = "not-needed"
# ID модели должен точно совпадать с тем, что возвращает эндпоинт /v1/models
MODEL = "qwen/qwen3.6-35b-a3b"

# Тип задачи для обработки текстов: "summarize", "extract_entities" или "classify"
TASK = "summarize"
# Входной CSV‑файл (должен содержать колонку text)
INPUT_FILE = "input.csv"
# Выходной CSV‑файл с результатами
OUTPUT_FILE = "output.csv"

# Параметры генерации для контроля «креативности» и разнообразия ответов
TEMPERATURE = 0.1   # Низкая температура — более предсказуемые, стабильные ответы
TOP_P = 0.5          # Top‑p сэмплирование ограничивает выборку токенов по вероятности
MAX_TOKENS = 150    # Лимит токенов на ответ — помогает ускорить генерацию и снизить таймауты

# Настройки повторных попыток при сетевых ошибках
MAX_RETRIES = 2      # Количество ретраев (не для таймаутов, а для разрывов соединения)
RETRY_DELAY = 5      # Задержка между попытками в секундах

# Таймаут в секундах (30 минут). Критически важен для больших моделей (35B) на слабых GPU:
# генерация может занимать несколько минут из‑за низкой скорости токенов в секунду.
TIMEOUT_SECONDS = 1800
# =============================================

# Инициализация клиента OpenAI с увеличенным таймаутом
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT_SECONDS)

def get_prompt(task: str) -> str:
    """
    Формирует системный промпт в зависимости от типа задачи.
    Промпты сделаны максимально лаконичными (без избыточного Chain‑of‑Thought),
    чтобы уменьшить время обработки промпта (TTFT) на моделях с малым VRAM.
    В каждом варианте:
      - чёткое требование вернуть только валидный JSON;
      - формат JSON;
      - один пример (Few‑shot);
      - указание «Текст для анализа:» в конце, чтобы модель понимала, где начинаются данные.
    """
    if task == "summarize":
        return """
Вы — генератор краткого резюме. Верните ТОЛЬКО валидный JSON‑объект. Без пояснений, без текста вне фигурных скобок.
Формат:
{
  "summary": "одно предложение с сутью текста",
  "keywords": ["ключевое слово 1", "ключевое слово 2"]
}
Пример (строго следуйте этому формату):
Текст: "Компания открыла новый завод в Сибири. Производство начнётся в октябре."
Ответ: {"summary": "Компания открывает новый завод в Сибири с запуском в октябре.", "keywords": ["завод", "Сибирь", "производство"]}
Текст для анализа:
"""
    elif task == "extract_entities":
        return """
Вы — система извлечения сущностей. Верните ТОЛЬКО корректный JSON‑объект. Без пояснений.
Формат:
{
  "persons": [],
  "organizations": [],
  "locations": [],
  "dates": []
}
Пример:
Текст: "Иван Петров из компании Рога и копыта приехал из Москвы вчера."
Ответ: {"persons": ["Иван Петров"], "organizations": ["Рога и копыта"], "locations": ["Москва"], "dates": ["вчера"]}
Текст для анализа:
"""
    elif task == "classify":
        return """
Вы — классификатор тональности. Верните ТОЛЬКО JSON. Без пояснений.
Формат:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}
Пример:
Текст: "Отличный сервис, всё быстро и удобно."
Ответ: {"sentiment": "positive", "confidence": 0.95}
Текст для анализа:
"""
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def read_csv(path: str) -> List[str]:
    """
    Читает CSV‑файл, пытаясь несколько кодировок (utf‑8‑sig, cp1251, utf‑8).
    Возвращает список строк из колонки 'text', пропуская пустые значения.
    Если файл не найден или колонка отсутствует — выбрасывает RuntimeError.
    """
    encodings = ["utf-8-sig", "cp1251", "utf-8"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None or "text" not in reader.fieldnames:
                    continue
                texts = [row["text"].strip() for row in reader if row.get("text", "").strip()]
                print(f"Прочитано текстов: {len(texts)} (кодировка {enc})")
                return texts
        except Exception:
            continue
    raise RuntimeError("Не удалось прочитать CSV ни с одной кодировкой.")

def call_with_retry(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    """
    Отправляет запрос к модели с возможностью повторной попытки при сетевых ошибках.

    Логика:
      - Ретраи делаются только при ConnectionError (разрыв соединения).
      - Таймауты (Timeout) не ретраятся — вместо этого используется большой глобальный timeout.
      - При других ошибках (HTTPError и т. п.) функция сразу возвращает None.
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
                print(f"[Попытка {attempt}/{MAX_RETRIES}] Ошибка соединения: {e}")
                if attempt < MAX_RETRIES:
                    print(f"Ждём {RETRY_DELAY} секунд перед следующей попыткой...")
                    time.sleep(RETRY_DELAY)
                continue
            else:
                # Другие ошибки (в т. ч. таймаут) не повторяем — считаем их критическими
                print(f"[Попытка {attempt}/{MAX_RETRIES}] Критическая ошибка: {e}")
                return None
    return None

def parse_json_strict(content: Optional[str]) -> Dict[str, Any]:
    """
    Строго парсит ответ модели в JSON.

    Особенности:
      - Удаляет возможные Markdown‑обёртки (```json ... ```), если модель их добавляет.
      - Возвращает словарь с полями:
          * data — распарсенные данные (dict) или None;
          * error — описание ошибки или None;
          * raw — исходный ответ модели.
    """
    result = {"data": None, "error": None, "raw": content or ""}
    if not content:
        result["error"] = "Нет ответа от модели"
        return result

    # Удаляем Markdown‑блоки, если они есть
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
            result["error"] = "JSON не является объектом (dict)"
    except json.JSONDecodeError as e:
        result["error"] = f"Невалидный JSON: {e}"
    return result

def main():
    # Читаем тексты из CSV
    texts = read_csv(INPUT_FILE)
    if not texts:
        print("Нет текстов для обработки.")
        return

    # Получаем нужный системный промпт
    system_prompt = get_prompt(TASK)
    results = []
    total_tokens = 0

    # Обрабатываем каждый текст
    for i, text in enumerate(texts, start=1):
        print(f"\n--- Обработка {i}/{len(texts)} ---")

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
            print(f"Ошибка: {error}")
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
        print(f"Успешно обработано: {i}")

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
    print(f"\nГотово. Результаты в {OUTPUT_FILE}")
    print(f"Обработано текстов: {len(results)}, примерно токенов: {total_tokens}")
if __name__ == "__main__":
    main()