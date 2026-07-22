import json
import csv
import re
from openai import OpenAI

# --- КОНФИГУРАЦИЯ ---
TIMEOUT_MINUTES = 60  # Таймаут для API запросов (можно менять: 5, 10, 20 минут и т.д.)
timeout_seconds = TIMEOUT_MINUTES * 60  # Конвертируем минуты в секунды (библиотека openai требует секунды)

# Создаем клиент для взаимодействия с LM Studio (локальный сервер)
client = OpenAI(
    base_url="http://192.168.8.11:1234/v1",  # URL локального сервера LM Studio
    api_key="lm-studio",  # Локальный API ключ (не используется, но требуется)
    timeout=timeout_seconds  # Устанавливаем таймаут в секундах
)

# Имя модели, которая загружена в LM Studio
MODEL_NAME = "qwen/qwen3.6-27b"


def get_zero_shot_prompt(task: str) -> str:
    """
    Функция возвращает промпт для Zero-shot обучения.
    Zero-shot - это подход, когда модель выполняет задачу без примеров, только на основе инструкции.
    
    Args:
        task (str): Тип задачи ("classify" - классификация тональности, "extract" - извлечение сущностей)
    
    Returns:
        str: Системный промпт для модели
    
    Raises:
        ValueError: Если передан неизвестный тип задачи
    """
    if task == "classify":
        # Промпт для классификации тональности текста
        return (
            "Ты классификатор тональности. Проанализируй текст и верни ТОЛЬКО валидный JSON. "
            "Никаких пояснений, никакого markdown. Только сырой JSON.\n"
            "Формат: {\"sentiment\": \"positive|negative|neutral\", \"confidence\": число от 0 до 1}"
        )
    elif task == "extract":
        # Промпт для извлечения именованных сущностей (PER, LOC, ORG)
        return (
            "Ты извлектель сущностей. Проанализируй текст и верни ТОЛЬКО валидный JSON. "
            "Никаких пояснений, никакого markdown. Только сырой JSON.\n"
            "Формат: {\"persons\": [\"Имя Фамилия\"], \"locations\": [\"Город\"], \"organizations\": [\"Название\"]}"
        )
    else:
        raise ValueError(f"Неизвестная задача: {task}")


def call_model(user_text: str, system_instruction: str) -> dict | None:
    """
    Отправляет запрос к модели и возвращает распарсенный результат.
    
    Args:
        user_text (str): Текст, который нужно обработать
        system_instruction (str): Инструкция для модели (промпт)
    
    Returns:
        dict | None: Результат в виде словаря (если успешен) или None (если ошибка)
    
    Process:
        1. Отправляет запрос к API с system и user сообщениями
        2. Извлекает ответ от модели
        3. Пытается извлечь JSON из ответа (убирает markdown-блоки и пояснения)
        4. Парсит JSON и возвращает результат
    """
    try:
        # Отправляем запрос к модели
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_instruction},  # Инструкция для модели
                {"role": "user", "content": user_text}  # Текст для обработки
            ],
            temperature=0.0,  # Детерминированный вывод (минимум креативности)
            top_p=1.0,  # Используем все возможные токены
            max_tokens=1024,  # Максимальное количество токенов в ответе
            stream=False  # Не используем потоковую передачу
        )

        # Извлекаем ответ из choices (openai v1.x+ API)
        choice = response.choices[0]
        content = choice.message.content
        
        # Если ответ пустой, возвращаем None
        if not content:
            return None
        
        # Пытаемся извлечь JSON из ответа (удаляем markdown-блоки и пояснения)
        json_str = content.strip()
        
        # Если ответ в markdown-блоках ```json ... ```, извлекаем только JSON
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Пытаемся найти JSON-объект в тексте (если модель добавила пояснения)
            json_match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', json_str)
            if json_match:
                json_str = json_match.group(1)
        
        # Парсим JSON строку в Python словарь
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Если JSON некорректный, возвращаем None
            return None
            
    except Exception as e:
        # В случае любой ошибки (сетевой, таймаут и т.д.) возвращаем None
        return None


def process_csv_batch(input_file: str, output_file: str, task: str):
    """
    Обрабатывает CSV файл с текстами и сохраняет результаты в JSON.
    
    Args:
        input_file (str): Путь к входному CSV файлу с текстами
        output_file (str): Путь к выходному JSON файлу для результатов
        task (str): Тип задачи ("classify" или "extract")
    
    Process:
        1. Читает CSV файл, пропуская заголовок
        2. Для каждой строки вызывает модель
        3. Сохраняет результаты в структурированный JSON
    """
    # Получаем системный промпт для текущей задачи
    system_prompt = get_zero_shot_prompt(task)
    
    results = []  # Список результатов для всех обработанных текстов
    
    # Открываем входной CSV файл
    with open(input_file, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        next(reader)  # Пропускаем заголовок CSV (первая строка)
        
        # Обрабатываем каждую строку
        for i, row in enumerate(reader):
            # Пропускаем пустые строки
            if not row or not row[0]:
                continue
            
            text = row[0]  # Берём первый столбец как текст
            result = call_model(text, system_prompt)  # Отправляем текст в модель
            
            # Формируем запись результата
            entry = {
                "id": i + 1,  # Порядковый номер
                "text": text,  # Исходный текст
                "prediction": result  # Результат от модели (или None если ошибка)
            }
            
            results.append(entry)
    
    # Формируем финальную структуру данных
    output_data = {
        "task": task,  # Тип задачи
        "model": MODEL_NAME,  # Использованная модель
        "total_count": len(results),  # Общее количество обработанных текстов
        "results": results  # Массив результатов
    }
    
    # Сохраняем результаты в JSON файл с форматированием
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)


# Точка входа в скрипт
if __name__ == "__main__":
    # Обработка input.csv и сохранение результатов в output.json
    # Задача: классификация тональности текстов
    process_csv_batch("input.csv", "output.json", "classify")
