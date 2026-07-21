import csv
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ================= НАСТРОЙКИ =================
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "not-needed"
MODEL = "qwen/qwen3.6-35b-a3b"

# Параметры генерации: низкая температура для стабильности, top_p ограничивает «разброс»
TEMPERATURE = 0.1
TOP_P = 0.5

TASK = "summarize"  # варианты: "summarize", "extract_entities", "classify"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"
MAX_RETRIES = 3
RETRY_DELAY = 2
# =============================================

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=60)

def get_prompt(task: str) -> str:
    """
    Возвращает промпт с Zero-shot + Few-shot + Chain-of-Thought.
    Модель получает чёткую роль, примеры (Few-shot), инструкцию по шагам (CoT)
    и строгое требование вернуть только валидный JSON.
    """
    # Chain-of-Thought для суммаризации
    cot_summarize = """
Chain-of-Thought (пошаговые рассуждения):
1. Прочитай входной текст целиком и выдели 3 самых важных факта: кто/что участвует, что произошло, где/когда это случилось.
2. На основе этих 3 фактов сформулируй одно предложение, которое передаёт главную суть текста (не больше 20–25 слов).
3. Извлеки из текста 3–5 ключевых слов, которые лучше всего описывают содержание (имена, названия, темы, локации).
4. Проверь, что в ключевых словах нет повторов и они действительно отражают суть.
5. Собери результат строго по шаблону JSON ниже, без каких‑либо комментариев до или после него.
6. Убедись, что внутри JSON нет Markdown-разметки (без ```json, без кавычек вокруг всего блока).
7. Если в тексте нет достаточной информации для выделения 3 фактов, сформулируй резюме на основе 1–2 самых значимых.
"""
    # Chain-of-Thought для извлечения сущностей
    cot_extract = """
Chain-of-Thought (пошаговые рассуждения):
1. Прочитай текст и последовательно пройдись по каждому предложению, отмечая упоминания людей. Собери их в список "persons" (только полные имена, без титулов и должностей).
2. Найди названия компаний, организаций, учреждений и добавь их в "organizations" (без общих слов вроде «компания», «фирма»).
3. Выпиши все локации (города, страны, улицы, здания) в список "locations".
4. Найди упоминания дат, дней недели, временных периодов и запиши их в "dates" в том виде, как они указаны в тексте (не перефразируй).
5. Проверь каждый элемент на соответствие своей категории; если есть сомнения — не добавляй.
6. Если для какой‑то категории сущностей не найдено, оставь соответствующий список пустым (например, "persons": []).
7. Сформируй итоговый JSON строго по шаблону, без пояснений и без Markdown-блоков.
8. Убедись, что все списки содержат только строки и не содержат вложенных структур.
"""
    # Chain-of-Thought для классификации тональности
    cot_classify = """
Chain-of-Thought (пошаговые рассуждения):
1. Прочитай текст и определи общий эмоциональный окрас: явно позитивный, явно негативный или нейтральный.
2. Обрати внимание на слова с эмоциональной окраской, усилители (очень, крайне, совершенно), отрицания и контекст.
3. Если в тексте есть противоречивые сигналы (и позитив, и негатив), оцени, какой из них доминирует по смыслу и силе выражения.
4. Определи уровень уверенности в своём выводе по шкале от 0.0 до 1.0:
   - 0.9–1.0 — однозначная тональность без противоречий;
   - 0.7–0.89 — преобладает один тон, но есть небольшие контраргументы;
   - 0.5–0.69 — смешанные сигналы, сложно однозначно отнести к одной категории;
   - ниже 0.5 — почти невозможно определить, либо тон очень слабый.
5. Запиши выбранную тональность как "positive", "negative" или "neutral".
6. Запиши confidence как число с плавающей точкой (например, 0.85).
7. Собери финальный JSON строго по шаблону, не добавляя никаких пояснений.
"""
    if task == "summarize":
        return f"""
Ты — строгий JSON-генератор для суммаризации. Твоя задача — вернуть ТОЛЬКО валидный JSON-объект без пояснений, без Markdown-блоков, без лишних слов.
{cot_summarize}
Few-shot (примеры):
Текст: «Компания открыла новый завод в Сибири. Производство начнётся в октябре. Ожидается 200 новых рабочих мест.»
Ответ: {{"summary": "Компания открывает новый завод в Сибири с запуском в октябре и созданием 200 рабочих мест.", "keywords": ["завод", "Сибирь", "производство", "рабочие места"]}}
Текст: «Конференция по ИИ прошла в Москве. Выступили 15 спикеров. Обсуждали этику ИИ и регулирование.»
Ответ: {{"summary": "В Москве прошла конференция по ИИ с участием 15 спикеров, где обсуждали этику и регулирование ИИ.", "keywords": ["ИИ", "конференция", "Москва", "этика", "регулирование"]}}
Формат ответа:
{{
  "summary": "одно предложение с сутью текста",
  "keywords": ["ключевое слово 1", "ключевое слово 2"]
}}
ВАЖНО: Не добавляй ни одного символа вне фигурных скобок. Только JSON.
"""
    elif task == "extract_entities":
        return f"""
Ты — система извлечения сущностей. Верни ТОЛЬКО корректный JSON-объект. Никаких пояснений.
{cot_extract}
Few-shot:
Текст: «Иван Петров из компании Рога и копыта приехал из Москвы вчера.»
Ответ: {{"persons": ["Иван Петров"], "organizations": ["Рога и копыта"], "locations": ["Москва"], "dates": ["вчера"]}}
Формат:
{{
  "persons": [],
  "organizations": [],
  "locations": [],
  "dates": []
}}
Только JSON, без пояснений.
"""
    elif task == "classify":
        return f"""
Ты — классификатор тональности. Верни ТОЛЬКО JSON. Без пояснений.
{cot_classify}
Few-shot:
Текст: «Отличный сервис, всё быстро и удобно.»
Ответ: {{"sentiment": "positive", "confidence": 0.95}}
Текст: «Ужасно медленно, ничего не работает.»
Ответ: {{"sentiment": "negative", "confidence": 0.98}}
Формат:
{{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0
}}
Только JSON.
"""
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def read_csv(path: str) -> List[str]:
    """Читает колонку 'text' из CSV, пробуя несколько кодировок."""
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
    """Простой retry с задержкой. Возвращает content или None при ошибке."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=200,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"[Попытка {attempt}/{MAX_RETRIES}] Ошибка: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return None


def parse_json_strict(content: Optional[str]) -> Dict[str, Any]:
    """
    Строгий парсинг JSON. Если не валидный JSON — возвращаем ошибку.
    Никаких регулярных выражений: либо чистый JSON, либо ошибка.
    """
    result = {"data": None, "error": None, "raw": content or ""}
    if not content:
        result["error"] = "Нет ответа от модели"
        return result
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            result["data"] = data
        else:
            result["error"] = "JSON не является объектом (dict)"
    except json.JSONDecodeError as e:
        result["error"] = f"Невалидный JSON: {e}"
    return result


def main():
    texts = read_csv(INPUT_FILE)
    if not texts:
        print("Нет текстов для обработки.")
        return
    system_prompt = get_prompt(TASK)
    results = []
    total_tokens = 0

    for i, text in enumerate(texts, start=1):
        print(f"\nОбработка {i}/{len(texts)}")
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Текст для анализа:\n\n{text}"},
        ]
        content = call_with_retry(messages)
        parsed = parse_json_strict(content)
        data = parsed["data"]
        error = parsed["error"]
        raw = parsed["raw"]

        # Учёт токенов: в этой простой версии берём «примерно» 200 токенов на запрос
        tokens_used = 200
        total_tokens += tokens_used
        row = {
            "id": i,
            "original": text,
            "tokens": tokens_used,
            "error": error or "",
            "raw_response": raw,
        }
        if error:
            # Заполняем пустые поля для согласованности CSV
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

        # Заполняем поля в зависимости от задачи
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

    # Определяем нужные колонки для CSV
    base_cols = ["id", "original", "tokens", "error", "raw_response"]
    if TASK == "summarize":
        fieldnames = base_cols + ["summary", "keywords"]
    elif TASK == "extract_entities":
        fieldnames = base_cols + ["persons", "organizations", "locations", "dates"]
    elif TASK == "classify":
        fieldnames = base_cols + ["sentiment", "confidence"]
    else:
        fieldnames = base_cols
    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nГотово. Результаты в {OUTPUT_FILE}")
    print(f"Обработано текстов: {len(results)}, примерно токенов: {total_tokens}")
if __name__ == "__main__":
    main()