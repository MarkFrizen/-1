import time
import json
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# --- КОНФИГУРАЦИЯ ---
# Убедитесь, что IP-адрес и порт совпадают с настройками LM Studio
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "lm-studio"
# Если модель 35B будет выдавать пустой ответ или висеть, замените на "qwen2.5-7b" или "qwen/qwen3.6-27b"
MODEL = "qwen/qwen3.6-35b-a3b"

# Параметры генерации
MAX_TOKENS = 500          # Лимит токенов
TEMPERATURE = 0.0         # Детерминированный вывод (важно для JSON)
TOP_P = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Инициализация клиента
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=3600)

def get_prompt(task: str) -> str:
    """
    Возвращает жесткий системный промпт.
    Никаких примеров (Few-shot), никаких вежливых вступлений.
    Только инструкция формата.
    """
    if task == "summarize":
        return (
            "Ты возвращаешь ТОЛЬКО валидный JSON объект. Никаких пояснений, "
            "никакого текста до или после JSON. Никаких markdown блоков (```json). "
            "Только сырой JSON. Формат: {\"summary\": \"краткое содержание\", \"keywords\": [\"слово1\"]}"
        )
    elif task == "extract_entities":
        return (
            "Ты возвращаешь ТОЛЬКО валидный JSON объект. Никаких пояснений. "
            "Никаких markdown блоков. Только сырой JSON. "
            "Формат: {\"persons\": [], \"organizations\": [], \"locations\": [], \"dates\": []}"
        )
    elif task == "classify":
        return (
            "Ты возвращаешь ТОЛЬКО валидный JSON объект. Никаких пояснений. "
            "Никаких markdown блоков. Только сырой JSON. "
            "Формат: {\"sentiment\": \"positive/negative/neutral\", \"confidence\": 0.0}"
        )
    else:
        raise ValueError(f"Неизвестная задача: {task}")

def call_with_retry(messages: List[ChatCompletionMessageParam]) -> Optional[str]:
    """
    Выполняет запрос к модели с повторными попытками при сетевых ошибках.
    Исправлена обработка структуры ответа для openai v1.x+.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Попытка {attempt}] Отправка запроса к модели {MODEL}...")

            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_tokens=MAX_TOKENS,
                stream=False,
                # Стоп-токены: если модель начнет писать новую строку после JSON, мы остановимся
                stop=["\n\n", "\n\n\n"]
            )

            # Проверка на наличие ответа
            if not resp.choices or len(resp.choices) == 0:
                print("[Ошибка] В ответе нет choices.")
                return None

            # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
            # Раньше здесь было: choice = resp.choices (это список)
            # Теперь берем первый элемент списка:
            choice = resp.choices

            # Проверка структуры ответа (защита от AttributeError)
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
            else:
                print("[Ошибка] Неожиданная структура ответа от API.")
                return None

            # Проверка на пустой контент (частая проблема с reasoning_content)
            if not content or content.strip() == "":
                print("[ВНИМАНИЕ] Поле 'content' пустое. Модель могла сгенерировать только рассуждения.")
                usage = resp.usage
                print(f"[Статистика токенов] Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}")
                return None
            print("[Успех] Ответ получен.")
            return content.strip()
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e)
            print(f"[Ошибка] Тип: {err_name}, Сообщение: {err_str}")

            # Логика повторных попыток только для сетевых проблем
            if "ConnectionError" in err_name or "connection" in err_str.lower() or "ReadTimeoutError" in err_name:
                if attempt < MAX_RETRIES:
                    print(f"Ожидание {RETRY_DELAY} сек перед повторной попыткой...")
                    time.sleep(RETRY_DELAY)
                    continue
            # Если это не сетевая ошибка или попытки кончились - выходим
            return None
    return None

def process_task(task_type: str, user_input: str) -> Optional[str]:
    """
    Основная функция обработки задачи.
    """
    system_prompt = get_prompt(task_type)
    messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    print(f"\n--- Запуск задачи: {task_type} ---")
    print(f"Входные данные: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
    response = call_with_retry(messages)
    if response:
        print(f"[Raw Response] {response}")
        # Попытка распарсить JSON для проверки валидности
        try:
            json_obj = json.loads(response)
            print("[Статус] Ответ является валидным JSON.")
            return response
        except json.JSONDecodeError:
            print("[Предупреждение] Ответ не является валидным JSON. Возможно, модель проигнорировала инструкции.")
            return response
    else:
        print("[FAIL] Не удалось получить валидный ответ от модели.")
        return None

if __name__ == "__main__":
    # --- ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ---
    # Пример 1: Классификация (быстрый тест)
    test_text = "Фильм был потрясающим, я смеялся и плакал одновременно."
    process_task("classify", test_text)
    # Пример 2: Извлечение сущностей (раскомментируйте для теста)
    # entity_text = "Иван Петров из компании ООО 'Вектор' встретился с Марией Сидоровой в Москве 15 мая."
    # process_task("extract_entities", entity_text)
    # Пример 3: Суммаризация (раскомментируйте для теста)
    # sum_text = "Сегодня был солнечный день. Мы пошли в парк. Там было много людей и собак."
    # process_task("summarize", sum_text)