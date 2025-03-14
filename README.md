# Advanced AI Assistant

Продвинутый ИИ-ассистент с возможностями голосового управления, распознавания речи, анализа текста и изображений.

## 🔧 Системные требования

- Windows 10/11
- Python 3.12
- Микрофон (для голосовых команд)
- Камера (для распознавания лиц, опционально)

## 📦 Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Обновите pip и установите базовые инструменты:
```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Установите зависимости:
```bash
pip install torch torchvision torchaudio
pip install tensorflow==2.18.1 tf-keras
pip install nltk pyaudio
pip install deepface==0.0.93
pip install -r requirements.txt
```

### 📋 Список зависимостей (requirements.txt):
```
pyautogui==0.9.54
opencv-python==4.8.1.78
numpy>=2.1.0
SpeechRecognition==3.10.0
pyttsx3==2.90
easyocr>=1.7.1
scikit-learn>=1.3.0
librosa>=0.10.1
deepface>=0.0.79
transformers>=4.35.2
torch>=2.2.0
PyPDF2>=3.0.1
python-docx>=1.0.1
nltk>=3.8.1
```

## 🚀 Запуск

1. Активируйте виртуальное окружение:
```bash
.\venv\Scripts\activate
```

2. Запустите программу:
```bash
python advanced_ai.py
```

## 💡 Возможности ИИ

### Голосовое управление
- Распознавание речи на русском и английском языках
- Синтез речи для ответов
- Голосовые команды для управления системой

### Работа с текстом
- Чтение и анализ текстовых файлов (.txt)
- Чтение PDF документов
- Чтение документов Word (.docx)
- Распознавание текста с экрана
- Анализ тональности текста
- Извлечение ключевой информации

### Компьютерное зрение
- Распознавание лиц
- Анализ эмоций
- Распознавание текста на изображениях

### Обучение
- Самообучение на основе диалогов
- Создание новых нейронных связей
- Адаптация к пользователю
- Запоминание успешных стратегий

### Базовые навыки
- Слушать и понимать команды
- Анализировать информацию
- Планировать действия
- Решать задачи
- Генерировать ответы

### Продвинутые навыки
- Создание контента
- Анализ данных
- Помощь в задачах
- Управление программами

## 📝 Основные команды

### Голосовые команды
- "Привет" - начать диалог
- "Слушай" - активировать режим прослушивания
- "Стоп" - остановить текущее действие
- "Читай [имя файла]" - чтение файла
- "Анализируй [текст]" - анализ текста
- "Помоги с [задача]" - получить помощь
- "Открой [программа]" - открыть программу
- "Закрой [программа]" - закрыть программу

### Команды для обучения
- "Учись [навык]" - начать изучение нового навыка
- "Запомни [информация]" - сохранить информацию
- "Покажи навыки" - показать список доступных навыков
- "Статистика" - показать статистику обучения

## 🛠️ Техническая информация

### Основные модули
1. ConsciousnessModule - модуль сознания
2. MemoryModule - модуль памяти
3. EmotionalIntelligenceModule - модуль эмоционального интеллекта
4. SelfLearningModule - модуль самообучения
5. EnvironmentModule - модуль взаимодействия с окружением
6. SelfCorrectionModule - модуль самокоррекции
7. PlanningModule - модуль планирования

### Нейронная сеть
- Сенсорные нейроны
- Моторные нейроны
- Ассоциативные нейроны
- Автоматическое создание новых связей

### Используемые технологии
- TensorFlow/Keras для нейронных сетей
- NLTK для обработки естественного языка
- PyAudio для работы с аудио
- OpenCV для компьютерного зрения
- DeepFace для анализа лиц
- Transformers для обработки текста

## 📦 Создание исполняемого файла

Для создания .exe файла используйте следующие команды:

```bash
pip install pyinstaller
pyinstaller --onefile --hidden-import=queue advanced_ai.py
```

Исполняемый файл будет создан в папке `dist`.

## ⚠️ Ограничения

- Не может выполнять физические действия
- Не имеет доступа к интернету (если не настроено)
- Не может устанавливать программы
- Не работает с конфиденциальными данными
- Требует настройки для специфических задач

## 🤝 Вклад в проект

Если вы хотите внести свой вклад в проект:
1. Сделайте форк репозитория
2. Создайте ветку для своих изменений
3. Внесите изменения
4. Создайте pull request

## 📄 Лицензия

MIT License 
