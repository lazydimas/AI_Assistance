# Advanced AI Assistant

Продвинутый ИИ-ассистент с возможностями голосового управления, распознавания речи, анализа текста и изображений.

[GitHub Repository](https://github.com/lazydimas/AI_Assistance)

## 🔧 Системные требования

### Windows
- Windows 10/11
- Python 3.12
- Микрофон (для голосовых команд)
- Камера (для распознавания лиц, опционально)
- Минимум 8 ГБ оперативной памяти
- Минимум 2 ГБ свободного места на диске

### Linux
- Ubuntu 20.04+ / Debian 11+ / CentOS 8+
- Python 3.12
- ALSA (для работы с аудио)
- Портативный микрофон или встроенный микрофон
- Камера (опционально)
- Минимум 8 ГБ оперативной памяти
- Минимум 2 ГБ свободного места на диске

## 📦 Пошаговая инструкция по установке

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/lazydimas/AI_Assistance.git
   cd AI_Assistance
   ```

2. **Создайте виртуальное окружение**
   ```bash
   python -m venv venv
   ```

3. **Активируйте виртуальное окружение**
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Обновите pip и установите базовые инструменты**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

5. **Установите системные зависимости**
   - Windows:
     ```bash
     # Не требуется дополнительных действий
     ```
   - Linux (Ubuntu/Debian):
     ```bash
     sudo apt-get update
     sudo apt-get install -y python3-dev python3-pip portaudio19-dev libespeak-dev
     sudo apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev
     ```
   - Linux (CentOS):
     ```bash
     sudo dnf install -y python3-devel portaudio-devel espeak-devel
     sudo dnf install -y ffmpeg-free libsm libXext libXrender-devel
     ```

6. **Установите основные зависимости**
   ```bash
   pip install torch torchvision torchaudio
   pip install tensorflow==2.18.1 tf-keras
   pip install nltk pyaudio
   pip install deepface==0.0.93
   ```

7. **Установите остальные зависимости**
   ```bash
   pip install -r requirements.txt
   ```

8. **Установите данные NLTK**
   ```bash
   python setup_nltk.py
   ```

## 🚀 Запуск проекта

1. **Убедитесь, что виртуальное окружение активировано**
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

2. **Запустите ИИ**
   ```bash
   python run_ai.py
   ```

## 📦 Создание исполняемого файла

### Windows (.exe)

1. **Установите PyInstaller**
   ```bash
   pip install pyinstaller
   ```

2. **Создайте .spec файл**
   ```bash
   pyinstaller --name advanced_ai --onefile ^
               --hidden-import=queue ^
               --hidden-import=pyttsx3.drivers ^
               --hidden-import=pyttsx3.drivers.sapi5 ^
               --hidden-import=numpy ^
               --hidden-import=tensorflow ^
               --hidden-import=torch ^
               --hidden-import=transformers ^
               --hidden-import=nltk ^
               --hidden-import=deepface ^
               --hidden-import=easyocr ^
               --add-data "venv/Lib/site-packages/nltk_data;nltk_data" ^
               run_ai.py
   ```

3. **Скомпилируйте проект**
   ```bash
   pyinstaller advanced_ai.spec
   ```

Исполняемый файл будет создан в папке `dist/advanced_ai.exe`

### Linux (.AppImage)

1. **Установите необходимые инструменты**
   ```bash
   pip install pyinstaller
   sudo apt-get install -y fuse
   ```

2. **Создайте .spec файл**
   ```bash
   pyinstaller --name advanced_ai --onefile \
               --hidden-import=queue \
               --hidden-import=pyttsx3.drivers \
               --hidden-import=pyttsx3.drivers.espeak \
               --hidden-import=numpy \
               --hidden-import=tensorflow \
               --hidden-import=torch \
               --hidden-import=transformers \
               --hidden-import=nltk \
               --hidden-import=deepface \
               --hidden-import=easyocr \
               --add-data "venv/lib/python3.12/site-packages/nltk_data:nltk_data" \
               run_ai.py
   ```

3. **Скомпилируйте проект**
   ```bash
   pyinstaller advanced_ai.spec
   ```

4. **Создайте AppImage (опционально)**
   ```bash
   # Установите appimagetool
   wget https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
   chmod +x appimagetool-x86_64.AppImage
   
   # Создайте AppImage
   ./appimagetool-x86_64.AppImage dist/advanced_ai
   ```

Исполняемый файл будет создан в папке `dist/advanced_ai` или `advanced_ai.AppImage`

## 🔍 Проверка установки

После запуска вы должны увидеть:
```
Инициализация ИИ...
ИИ успешно инициализирован
Запуск основного цикла...
```

## ❗ Решение возможных проблем

1. **Ошибка "No module named..."**
   ```bash
   pip install [имя-отсутствующего-модуля]
   ```

2. **Ошибка с PyAudio**
   - Windows:
     ```bash
     pip install pipwin
     pipwin install pyaudio
     ```

3. **Ошибка с NLTK данными**
   ```bash
   python setup_nltk.py
   ```

4. **Проблемы с микрофоном**
   - Проверьте, что микрофон подключен и работает
   - Проверьте права доступа к микрофону

5. **Проблемы с камерой**
   - Проверьте подключение камеры
   - Дайте разрешение на использование камеры

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

## 📁 Структура проекта

```
project/
├── advanced_ai.py      # Основной файл ИИ
├── run_ai.py          # Файл для запуска
├── setup_nltk.py      # Установка данных NLTK
├── requirements.txt   # Зависимости проекта
├── README.md         # Документация
└── neurons/          # Модули нейронной сети
```

## 🛠️ Техническая информация

### Основные модули
1. ConsciousnessModule - модуль сознания
2. MemoryModule - модуль памяти
3. EmotionalIntelligenceModule - модуль эмоционального интеллекта
4. SelfLearningModule - модуль самообучения
5. EnvironmentModule - модуль взаимодействия с окружением
6. SelfCorrectionModule - модуль самокоррекции
7. PlanningModule - модуль планирования

### Используемые технологии
- TensorFlow/Keras для нейронных сетей
- NLTK для обработки естественного языка
- PyAudio для работы с аудио
- OpenCV для компьютерного зрения
- DeepFace для анализа лиц
- Transformers для обработки текста

## ⚠️ Ограничения

- Не может выполнять физические действия
- Не имеет доступа к интернету (если не настроено)
- Не может устанавливать программы
- Не работает с конфиденциальными данными
- Требует настройки для специфических задач

## 🤝 Поддержка

Если у вас возникли проблемы:
1. Проверьте раздел "Решение возможных проблем"
2. Создайте issue в репозитории
3. Свяжитесь с разработчиками

## 📄 Лицензия

MIT License 
