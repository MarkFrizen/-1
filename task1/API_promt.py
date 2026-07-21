import time
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# --- КОНФИГУРАЦИЯ ---
BASE_URL = "http://192.168.8.11:1234/v1"
API_KEY = "lm-studio"  # Стандартный ключ для LM Studio
MODEL = "qwen/qwen3.6-35b-a3b"  # Из вашего списка моделей

# Параметры генерации
MAX_TOKENS = 500          # Увеличено с 150, чтобы модель успела написать ответ после рассуждений
TEMPERATURE = 0.0         # Детерминированный ответ (важно для JSON)
TOP_P = 1.0
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Инициализация клиента
client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=3600)

def get_prompt(task: str) -> str:
    """
    Возвращает максимально жесткий системный промпт.
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
                stop=["\n\n", "\n\n\n"]
            )
            # --- ИСПРАВЛЕНИЕ ЗДЕСЬ ---
            # В новых версиях openai resp.choices - это список объектов CompletionChoice
            if not resp.choices or len(resp.choices) == 0:
                print("[Ошибка] В ответе нет choices.")
                return None
            # Получаем первый выбор
            choice = resp.choices
            # В новых версиях у choice есть атрибут .message, который является объектом ChatCompletionMessage
            # Но для безопасности проверяем наличие атрибута
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                content = choice.message.content
            else:
                # Фолбэк: если структура странная, пробуем достать контент напрямую (редко, но бывает)
                content = getattr(choice, 'content', None)
            if not content or content.strip() == "":
                print("[ВНИМАНИЕ] Поле 'content' пустое. Модель могла сгенерировать только рассуждения.")
                usage = resp.usage
                print(f"[Статистика] Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}")
                return None
            print("[Успех] Ответ получен.")
            return content.strip()
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e)
            print(f"[Ошибка] Тип: {err_name}, Сообщение: {err_str}")
            if "ConnectionError" in err_name or "connection" in err_str.lower() or "ReadTimeoutError" in err_name:
                if attempt < MAX_RETRIES:
                    print(f"Ожидание {RETRY_DELAY} сек перед повторной попыткой...")
                    time.sleep(RETRY_DELAY)
                    continue
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
        # Здесь можно добавить парсинг JSON через json.loads(response)
        return response
    else:
        print("[FAIL] Не удалось получить валидный ответ от модели.")
        return None
if __name__ == "__main__":
    # --- ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ---
    # Пример 1: Классификация (быстрый тест)
    # Если модель 35B все еще "думает" слишком долго и съедает токены,
    # этот тест покажет, есть ли вообще ответ в поле content.
    test_text = "Фильм был потрясающим, я смеялся и плакал одновременно."
    process_task("classify", test_text)
    # Пример 2: Извлечение сущностей (требует больше токенов)
    # entity_text = "Иван Петров из компании ООО 'Вектор' встретился с Марией Сидоровой в Москве 15 мая."
    # process_task("extract_entities", entity_text)
    # Пример 3: Суммаризация
    # sum_text = "Сегодня был солнечный день. Мы пошли в парк. Там было много людей и собак."
    # process_task("summarize", sum_text)