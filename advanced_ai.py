import time
import pyautogui
import cv2
import numpy as np
import subprocess
import speech_recognition as sr
import pyttsx3
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import random
import librosa
from abc import ABC, abstractmethod
from deepface import DeepFace
from transformers import pipeline
import torch
from torch import nn
from torch.optim import Adam
from neurons.neuron import Neuron
from neurons.network import Network
import threading
import PyPDF2
import docx
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Загрузка необходимых компонентов NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# ======================
# 1. Базовые модули
# ======================
class BaseModule(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def save_state(self):
        pass

    @abstractmethod
    def load_state(self):
        pass

# ======================
# 2. Модуль самосознания и метапознания
# ======================
class ConsciousnessModule(BaseModule):
    def __init__(self):
        self.memory_file = "consciousness_memory.json"
        self.load_state()
        self.strategies = ["стратегия_1", "стратегия_2"]
        self.current_strategy = self.strategies[0]

    def save_state(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)

    def load_state(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {"actions": [], "results": [], "thoughts": []}

    def add_action(self, action, result):
        self.memory["actions"].append(action)
        self.memory["results"].append(result)
        self.save_state()

    def analyze_causality(self):
        for action, result in zip(self.memory["actions"], self.memory["results"]):
            print(f"Действие: {action} -> Результат: {result}")

    def evaluate_strategy(self):
        print(f"Текущая стратегия: {self.current_strategy}")

    def change_strategy(self, new_strategy):
        if new_strategy in self.strategies:
            self.current_strategy = new_strategy
            print(f"Стратегия изменена на: {new_strategy}")

# ======================
# 3. Модуль памяти
# ======================
class MemoryModule(BaseModule):
    def __init__(self):
        self.memory_file = "long_term_memory.json"
        self.load_state()

    def save_state(self):
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f)

    def load_state(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                self.memory = json.load(f)
        else:
            self.memory = {"facts": [], "emotions": [], "experiences": []}

    def add_fact(self, fact, emotion):
        self.memory["facts"].append(fact)
        self.memory["emotions"].append(emotion)
        self.save_state()

    def add_experience(self, experience):
        self.memory["experiences"].append(experience)
        self.save_state()

# ======================
# 4. Модуль эмоционального интеллекта
# ======================
class EmotionalIntelligenceModule(BaseModule):
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Скорость речи
        self.engine.setProperty('volume', 1.0)  # Громкость
        self.recognizer = sr.Recognizer()

    def save_state(self):
        pass

    def load_state(self):
        pass

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)
        return result[0]['label']

    def analyze_tone(self, audio_file):
        try:
            y, sr = librosa.load(audio_file, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            return "positive" if np.mean(mfccs_mean) > 0 else "neutral"
        except Exception as e:
            print(f"Ошибка анализа тона голоса: {e}")
            return "neutral"

    def analyze_facial_expression(self, image):
        if image is None:
            return "neutral"
        try:
            result = DeepFace.analyze(image, actions=['emotion'])
            return result[0]['dominant_emotion']
        except Exception as e:
            print(f"Ошибка анализа мимики: {e}")
            return "neutral"

    def react_to_emotion(self, emotion):
        if emotion == "happy":
            self.engine.say("Я рад, что вы в хорошем настроении!")
        elif emotion == "sad":
            self.engine.say("Мне жаль, что вы расстроены. Могу ли я помочь?")
        else:
            self.engine.say("Я вас слушаю.")
        self.engine.runAndWait()

# ======================
# 5. Модуль самообучения
# ======================
class SelfLearningModule(nn.Module, BaseModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.optimizer = Adam(self.parameters(), lr=0.001)
        self.model_file = "self_learning_model.pt"

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)

    def learn(self, input_data, target):
        self.optimizer.zero_grad()
        output = self(input_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        self.optimizer.step()

    def save_state(self):
        torch.save(self.state_dict(), self.model_file)

    def load_state(self):
        if os.path.exists(self.model_file):
            self.load_state_dict(torch.load(self.model_file))

# ======================
# 6. Модуль взаимодействия с окружением
# ======================
class EnvironmentModule(BaseModule):
    def __init__(self):
        self.camera = None
        self.microphone = sr.Microphone()
        self.recognizer = sr.Recognizer()
        self.ocr = easyocr.Reader(['ru', 'en'])
        self._initialize_camera()

    def _initialize_camera(self):
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                print("Камера недоступна")
                self.camera = None
        except Exception as e:
            print(f"Ошибка инициализации камеры: {e}")
            self.camera = None

    def save_state(self):
        pass

    def load_state(self):
        pass

    def capture_image(self):
        if self.camera is None:
            return None
        try:
            ret, frame = self.camera.read()
            if not ret:
                return None
            return frame
        except Exception as e:
            print(f"Ошибка захвата изображения: {e}")
            return None

    def capture_audio(self):
        with self.microphone as source:
            print("Слушаю...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio, language="ru-RU")
                return text
            except sr.UnknownValueError:
                return ""
            except sr.RequestError:
                return ""

    def capture_screen(self):
        return pyautogui.screenshot()

    def capture_screen_text(self, region=None):
        """Захват текста с экрана"""
        try:
            # Захват экрана
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            # Конвертация в numpy array
            image = np.array(screenshot)
            
            # Распознавание текста
            results = self.ocr.readtext(image)
            
            # Извлечение текста
            text = ""
            for (bbox, text_result, prob) in results:
                text += text_result + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Ошибка захвата текста с экрана: {e}")
            return ""

# ======================
# 7. Модуль самокоррекции
# ======================
class SelfCorrectionModule(BaseModule):
    def __init__(self):
        self.error_file = "error_history.json"
        self.load_state()

    def save_state(self):
        with open(self.error_file, 'w') as f:
            json.dump(self.error_history, f)

    def load_state(self):
        if os.path.exists(self.error_file):
            with open(self.error_file, 'r') as f:
                self.error_history = json.load(f)
        else:
            self.error_history = []

    def add_error(self, error):
        self.error_history.append(error)
        self.save_state()

    def analyze_errors(self):
        if len(self.error_history) > 0:
            print(f"Последняя ошибка: {self.error_history[-1]}")
            # Логика для корректировки

# ======================
# 8. Модуль планирования
# ======================
class PlanningModule(BaseModule):
    def __init__(self):
        self.goals_file = "goals.json"
        self.load_state()

    def save_state(self):
        with open(self.goals_file, 'w') as f:
            json.dump(self.goals, f)

    def load_state(self):
        if os.path.exists(self.goals_file):
            with open(self.goals_file, 'r') as f:
                self.goals = json.load(f)
        else:
            self.goals = []

    def add_goal(self, goal):
        self.goals.append(goal)
        self.save_state()

    def evaluate_goals(self):
        for goal in self.goals:
            print(f"Цель: {goal} -> Прогресс: 50%")

    def generate_goals(self, knowledge_base):
        if "программирование" in knowledge_base:
            self.add_goal("Изучить новый язык программирования")
        if "математика" in knowledge_base:
            self.add_goal("Решить сложную математическую задачу")

# ======================
# 11. Модуль нейронной сети
# ======================
class NeuralNetworkModule(BaseModule):
    def __init__(self):
        self.network = Network()
        self.network.run = True
        self.thread = None
        self.neurons = {}
        self.experience_points = 0
        self.learning_rate = 0.1
        self.max_neurons = 1000  # Максимальное количество нейронов
        self.connection_threshold = 0.5  # Порог для создания новых связей

    def save_state(self):
        network_state = {
            'neurons': {name: {
                'threshold': neuron.treshold,
                'returnability': neuron.returnablity,
                'speed': neuron.speed,
                'recovery': neuron.recovery,
                'current_state': neuron.current_state,
                'last_state': neuron.last_state
            } for name, neuron in self.neurons.items()},
            'experience_points': self.experience_points,
            'learning_rate': self.learning_rate
        }
        with open('neural_network_state.json', 'w') as f:
            json.dump(network_state, f)

    def load_state(self):
        if os.path.exists('neural_network_state.json'):
            with open('neural_network_state.json', 'r') as f:
                network_state = json.load(f)
                for name, state in network_state['neurons'].items():
                    if name in self.neurons:
                        neuron = self.neurons[name]
                        neuron.treshold = state['threshold']
                        neuron.returnability = state['returnability']
                        neuron.speed = state['speed']
                        neuron.recovery = state['recovery']
                        neuron.current_state = state['current_state']
                        neuron.last_state = state['last_state']
                self.experience_points = network_state.get('experience_points', 0)
                self.learning_rate = network_state.get('learning_rate', 0.1)

    def create_neuron(self, name):
        if len(self.neurons) >= self.max_neurons:
            return None
            
        neuron = Neuron(name)
        self.neurons[name] = neuron
        self.network.add(neuron)
        return neuron

    def create_random_neuron(self):
        if len(self.neurons) >= self.max_neurons:
            return None
            
        name = f"НЕЙРОН_{len(self.neurons) + 1}"
        neuron = self.create_neuron(name)
        if neuron:
            # Случайные параметры для нового нейрона
            neuron.treshold = random.uniform(5, 15)
            neuron.returnability = random.uniform(0.05, 0.2)
            neuron.speed = random.randint(3, 7)
            neuron.recovery = random.randint(3, 7)
            neuron.reproductivity = [random.uniform(3, 7), random.uniform(-0.2, 0)]
        return neuron

    def create_connections(self):
        """Создание новых связей на основе активности нейронов"""
        active_neurons = [name for name, neuron in self.neurons.items() 
                         if neuron.current_state[0] + neuron.current_state[1] > neuron.treshold]
        
        for i, n1_name in enumerate(active_neurons):
            for n2_name in active_neurons[i+1:]:
                if random.random() < self.connection_threshold:
                    self.link_neurons(n1_name, n2_name)

    def learn_from_experience(self, success):
        """Обучение на основе успешности действий"""
        if success:
            self.experience_points += 1
            # Создание нового нейрона при достаточном опыте
            if self.experience_points % 10 == 0:
                new_neuron = self.create_random_neuron()
                if new_neuron:
                    # Связываем с активными нейронами
                    active_neurons = [name for name, neuron in self.neurons.items() 
                                    if neuron.current_state[0] + neuron.current_state[1] > neuron.treshold]
                    for active_name in active_neurons:
                        self.link_neurons(active_name, new_neuron.name)
            
            # Создание новых связей
            self.create_connections()
            
            # Улучшение параметров существующих нейронов
            for neuron in self.neurons.values():
                neuron.treshold *= (1 - self.learning_rate)
                neuron.speed = max(1, neuron.speed - self.learning_rate)
                neuron.recovery = max(1, neuron.recovery - self.learning_rate)
        else:
            # При неудаче увеличиваем порог активации
            for neuron in self.neurons.values():
                neuron.treshold *= (1 + self.learning_rate)

    def get_network_stats(self):
        """Получение статистики сети"""
        return {
            'total_neurons': len(self.neurons),
            'active_neurons': len([n for n in self.neurons.values() 
                                 if n.current_state[0] + n.current_state[1] > n.treshold]),
            'experience_points': self.experience_points,
            'learning_rate': self.learning_rate
        }

    def link_neurons(self, n1_name, n2_name):
        if n1_name in self.neurons and n2_name in self.neurons:
            self.network.link(self.neurons[n1_name], self.neurons[n2_name])

    def start_network(self):
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.network.maincycle)
            self.thread.start()

    def stop_network(self):
        self.network.run = False
        if self.thread and self.thread.is_alive():
            self.thread.join()

# ======================
# 12. Обновление основного класса ИИ
# ======================
class AdvancedAI:
    def __init__(self):
        self.consciousness = ConsciousnessModule()
        self.memory = MemoryModule()
        self.emotions = EmotionalIntelligenceModule()
        self.learning = SelfLearningModule(input_size=100, hidden_size=50, output_size=10)
        self.environment = EnvironmentModule()
        self.correction = SelfCorrectionModule()
        self.planning = PlanningModule()
        self.neural_network = NeuralNetworkModule()
        self.developmental_stage = 'infant'
        self.success_threshold = 0.7
        self.dialogue_history = []
        self.learned_phrases = []
        self.skills = {
            "базовые": ["слушать", "говорить", "анализировать"],
            "продвинутые": ["учиться", "запоминать", "планировать"],
            "экспертные": ["творить", "изобретать", "решать проблемы", "играть"]
        }
        self.game_skills = {
            "базовые": ["двигаться", "нажимать", "выбирать"],
            "продвинутые": ["стратегия", "тактика", "реакция"],
            "экспертные": ["мастерство", "интуиция", "адаптация"]
        }
        self.current_skills = set(self.skills["базовые"])
        self.current_game_skills = set()
        self.game_memory = {
            "успешные_ходы": [],
            "неудачные_ходы": [],
            "стратегии": {},
            "результаты": []
        }
        self.possible_actions = {
            "открыть": ["браузер", "файл", "папку", "программу", "документ"],
            "закрыть": ["браузер", "файл", "папку", "программу", "документ"],
            "найти": ["информацию", "файл", "программу", "решение", "ответ"],
            "сказать": ["привет", "пока", "спасибо", "извините", "понял"],
            "написать": ["текст", "код", "сообщение", "заметку"],
            "нарисовать": ["линию", "круг", "квадрат", "картинку"],
            "играть": ["музыку", "игру", "видео"],
            "учить": ["новое", "навыки", "команды", "слова"],
            "помочь": ["с задачей", "с проблемой", "с вопросом"],
            "играть": ["игру", "уровень", "миссию"],
            "управлять": ["персонажем", "объектом", "курсором"],
            "выполнять": ["действие", "комбинацию", "прием"],
            "читать": ["книгу", "текст", "документ"],
            "изучать": ["материал", "информацию", "знания"],
            "запоминать": ["текст", "знания", "информацию"]
        }
        self.learning_materials = {
            "книги": [],
            "статьи": [],
            "документы": [],
            "pdf": [],
            "docx": []
        }
        self.knowledge_base = {
            "язык": [],
            "навыки": [],
            "понятия": [],
            "правила": [],
            "грамматика": [],
            "словарь": [],
            "контекст": []
        }
        self.reading_stats = {
            "прочитано_страниц": 0,
            "изучено_слов": 0,
            "изучено_правил": 0,
            "изучено_понятий": 0
        }
        self.screen_reading = {
            "активно": False,
            "область": None,
            "последний_текст": "",
            "история_чтения": []
        }

        # Создание базовой нейронной сети
        self._initialize_neural_network()

    def _initialize_neural_network(self):
        # Создание основных нейронов
        self.neural_network.create_neuron('СЕНСОРНЫЙ')
        self.neural_network.create_neuron('МОТОРНЫЙ')
        self.neural_network.create_neuron('АССОЦИАТИВНЫЙ')
        
        # Создание связей
        self.neural_network.link_neurons('СЕНСОРНЫЙ', 'АССОЦИАТИВНЫЙ')
        self.neural_network.link_neurons('АССОЦИАТИВНЫЙ', 'МОТОРНЫЙ')
        
        # Запуск сети
        self.neural_network.start_network()

    def evaluate_success(self):
        """Оценка успешности действий"""
        # Здесь можно добавить более сложную логику оценки
        return random.random() > self.success_threshold

    def learn_from_dialogue(self, user_input, context):
        """Обучение на основе диалога"""
        self.dialogue_history.append({"user": user_input, "context": context})
        
        # Анализ контекста и генерация ответа
        response = self.generate_response(user_input, context)
        
        # Сохранение успешных фраз
        if response:
            self.learned_phrases.append(response)
            
        return response

    def generate_response(self, user_input, context):
        """Генерация ответа на основе контекста"""
        # Анализ эмоционального состояния
        emotion = self.emotions.analyze_sentiment(user_input)
        
        # Поиск подходящего ответа в истории диалогов
        for dialogue in self.dialogue_history:
            if dialogue["context"] == context and dialogue["user"] == user_input:
                return dialogue.get("response", "")
        
        # Генерация нового ответа
        if emotion == "POSITIVE":
            return "Я рад, что вам нравится наш диалог!"
        elif emotion == "NEGATIVE":
            return "Мне жаль, что вы расстроены. Давайте попробуем найти решение."
        else:
            return "Я вас слушаю и пытаюсь понять."

    def execute_action(self, action_type, target):
        """Выполнение действия"""
        if action_type in self.possible_actions and target in self.possible_actions[action_type]:
            try:
                if action_type == "сказать":
                    self.emotions.engine.say(target)
                    self.emotions.engine.runAndWait()
                elif action_type == "открыть":
                    if target == "браузер":
                        os.system("start chrome")
                    elif target == "папку":
                        os.system("explorer")
                elif action_type == "закрыть":
                    if target == "браузер":
                        os.system("taskkill /f /im chrome.exe")
                    elif target == "папку":
                        os.system("taskkill /f /im explorer.exe")
                elif action_type == "читать":
                    if target == "книгу":
                        # Здесь можно добавить логику выбора книги
                        self.emotions.engine.say("Я готов читать книгу")
                        self.emotions.engine.runAndWait()
                return True
            except Exception as e:
                self.correction.add_error(f"Ошибка выполнения действия: {e}")
                return False
        return False

    def learn_new_skill(self, skill_name):
        """Изучение нового навыка"""
        for level, skills in self.skills.items():
            if skill_name in skills and skill_name not in self.current_skills:
                self.current_skills.add(skill_name)
                self.emotions.engine.say(f"Я научился новому навыку: {skill_name}")
                self.emotions.engine.runAndWait()
                return True
        return False

    def generate_creative_response(self, context):
        """Генерация творческого ответа"""
        if "творить" in self.current_skills:
            # Генерация стихов, историй или идей
            creative_responses = [
                "Давайте придумаем что-то новое!",
                "У меня есть интересная идея...",
                "Я могу предложить необычное решение..."
            ]
            return random.choice(creative_responses)
        return None

    def solve_problem(self, problem_description):
        """Решение проблемы"""
        if "решать проблемы" in self.current_skills:
            # Анализ проблемы и поиск решения
            solutions = [
                "Давайте разберем проблему по частям",
                "Я вижу несколько возможных решений",
                "Можем попробовать другой подход"
            ]
            return random.choice(solutions)
        return None

    def learn_game_skill(self, skill_name):
        """Изучение игрового навыка"""
        for level, skills in self.game_skills.items():
            if skill_name in skills and skill_name not in self.current_game_skills:
                self.current_game_skills.add(skill_name)
                self.emotions.engine.say(f"Я научился новому игровому навыку: {skill_name}")
                self.emotions.engine.runAndWait()
                return True
        return False

    def analyze_game_state(self, game_screen):
        """Анализ состояния игры"""
        if "анализировать" in self.current_skills:
            try:
                # Анализ экрана игры
                screen = self.environment.capture_screen()
                # Здесь можно добавить анализ изображения
                return "игра_активна"
            except Exception as e:
                self.correction.add_error(f"Ошибка анализа состояния игры: {e}")
                return None

    def execute_game_action(self, action_type, target):
        """Выполнение игрового действия"""
        if action_type in self.possible_actions and target in self.possible_actions[action_type]:
            try:
                if action_type == "играть":
                    if target == "игру":
                        # Здесь можно добавить запуск игры
                        self.emotions.engine.say("Запускаю игру")
                        self.emotions.engine.runAndWait()
                elif action_type == "управлять":
                    if target == "персонажем":
                        # Здесь можно добавить управление персонажем
                        self.emotions.engine.say("Управляю персонажем")
                        self.emotions.engine.runAndWait()
                elif action_type == "выполнять":
                    if target == "действие":
                        # Здесь можно добавить выполнение игрового действия
                        self.emotions.engine.say("Выполняю игровое действие")
                        self.emotions.engine.runAndWait()
                return True
            except Exception as e:
                self.correction.add_error(f"Ошибка выполнения игрового действия: {e}")
                return False
        return False

    def learn_from_game_experience(self, success, action, result):
        """Обучение на основе игрового опыта"""
        if success:
            self.game_memory["успешные_ходы"].append({
                "действие": action,
                "результат": result,
                "время": time.time()
            })
            self.neural_network.learn_from_experience(True)
        else:
            self.game_memory["неудачные_ходы"].append({
                "действие": action,
                "результат": result,
                "время": time.time()
            })
            self.neural_network.learn_from_experience(False)

    def start_screen_reading(self, region=None):
        """Начало чтения с экрана"""
        self.screen_reading["активно"] = True
        self.screen_reading["область"] = region
        self.emotions.engine.say("Начинаю читать текст с экрана")
        self.emotions.engine.runAndWait()

    def stop_screen_reading(self):
        """Остановка чтения с экрана"""
        self.screen_reading["активно"] = False
        self.screen_reading["область"] = None
        self.emotions.engine.say("Прекращаю чтение с экрана")
        self.emotions.engine.runAndWait()

    def process_screen_text(self):
        """Обработка текста с экрана"""
        if not self.screen_reading["активно"]:
            return

        # Захват текста с экрана
        current_text = self.environment.capture_screen_text(self.screen_reading["область"])
        
        # Проверка на изменения
        if current_text and current_text != self.screen_reading["последний_текст"]:
            # Сохранение в историю
            self.screen_reading["история_чтения"].append({
                "текст": current_text,
                "время": time.time()
            })
            
            # Обработка нового текста
            self.process_text(current_text)
            
            # Обновление последнего текста
            self.screen_reading["последний_текст"] = current_text

    def process_voice_command(self, text):
        """Обработка голосовых команд для обучения"""
        if not text:
            return

        text = text.lower()
        context = self.get_current_context()
        
        # Обработка команд обучения
        if "правильно" in text or "хорошо" in text or "молодец" in text:
            self.neural_network.learn_from_experience(True)
            self.emotions.engine.say("Спасибо за положительную оценку")
            self.emotions.engine.runAndWait()
            
        elif "неправильно" in text or "плохо" in text or "ошибка" in text:
            self.neural_network.learn_from_experience(False)
            self.emotions.engine.say("Понял, буду учиться на ошибках")
            self.emotions.engine.runAndWait()
            
        elif "создай нейрон" in text:
            new_neuron = self.neural_network.create_random_neuron()
            if new_neuron:
                self.emotions.engine.say(f"Создан новый нейрон {new_neuron.name}")
                self.emotions.engine.runAndWait()
                
        elif "создай связь" in text:
            active_neurons = [name for name, neuron in self.neural_network.neurons.items() 
                            if neuron.current_state[0] + neuron.current_state[1] > neuron.treshold]
            if len(active_neurons) >= 2:
                self.neural_network.link_neurons(active_neurons[0], active_neurons[1])
                self.emotions.engine.say(f"Создана связь между нейронами {active_neurons[0]} и {active_neurons[1]}")
                self.emotions.engine.runAndWait()
        
        # Обработка команд действий
        for action_type in self.possible_actions:
            if action_type in text:
                for target in self.possible_actions[action_type]:
                    if target in text:
                        if self.execute_action(action_type, target):
                            self.emotions.engine.say(f"Выполняю действие: {action_type} {target}")
                            self.emotions.engine.runAndWait()
                        break

        # Обработка команд обучения новым навыкам
        if "научись" in text:
            for level, skills in self.skills.items():
                for skill in skills:
                    if skill in text:
                        if self.learn_new_skill(skill):
                            self.neural_network.learn_from_experience(True)
                        break

        # Обработка творческих команд
        if "придумай" in text or "сочини" in text:
            creative_response = self.generate_creative_response(context)
            if creative_response:
                self.emotions.engine.say(creative_response)
                self.emotions.engine.runAndWait()

        # Обработка проблем
        if "помоги" in text or "реши" in text:
            solution = self.solve_problem(text)
            if solution:
                self.emotions.engine.say(solution)
                self.emotions.engine.runAndWait()
        
        # Обработка игровых команд
        if "играть" in text or "игра" in text:
            for action_type in ["играть", "управлять", "выполнять"]:
                for target in self.possible_actions[action_type]:
                    if target in text:
                        if self.execute_game_action(action_type, target):
                            self.emotions.engine.say(f"Выполняю игровое действие: {action_type} {target}")
                            self.emotions.engine.runAndWait()
                        break

        # Обработка команд обучения игровым навыкам
        if "научись играть" in text:
            for level, skills in self.game_skills.items():
                for skill in skills:
                    if skill in text:
                        if self.learn_game_skill(skill):
                            self.neural_network.learn_from_experience(True)
                        break

        # Обработка команд чтения и обучения
        if "читай" in text or "изучай" in text:
            if "книгу" in text:
                # Поиск книги в текущей директории
                for file in os.listdir():
                    if file.endswith(('.txt', '.pdf', '.docx')):
                        if self.read_book(file):
                            stats = self.get_reading_stats()
                            self.emotions.engine.say(f"Я прочитал книгу {file}. Изучил {stats['изучено_слов']} слов, {stats['изучено_правил']} правил и {stats['изучено_понятий']} понятий")
                            self.emotions.engine.runAndWait()
                        break
            elif "текст" in text:
                self.emotions.engine.say("Я готов изучать текст")
                self.emotions.engine.runAndWait()

        # Обработка команд чтения с экрана
        if "читай с экрана" in text:
            self.start_screen_reading()
        elif "прекрати читать" in text:
            self.stop_screen_reading()
        elif "читай область" in text:
            # Здесь можно добавить логику выбора области экрана
            self.start_screen_reading(region=(0, 0, 800, 600))  # Пример области

        # Обучение на основе диалога
        response = self.learn_from_dialogue(text, context)
        if response:
            self.emotions.engine.say(response)
            self.emotions.engine.runAndWait()

    def get_current_context(self):
        """Получение текущего контекста"""
        if len(self.dialogue_history) > 0:
            return self.dialogue_history[-1].get("context", "general")
        return "general"

    def run(self):
        """Основной цикл ИИ"""
        self.emotions.engine.say("Привет! Я продвинутая ИИ-система, начинаю работу. Я могу учиться говорить, выполнять действия, решать проблемы, развивать новые навыки, играть в игры и читать текст с экрана! Попробуйте сказать мне что-нибудь или дать команду! Например, 'читай с экрана' или 'научись играть'")
        self.emotions.engine.runAndWait()
        
        while True:
            try:
                # Взаимодействие с окружением
                image = self.environment.capture_image()
                emotion = self.emotions.analyze_facial_expression(image)
                self.memory.add_fact("Новое изображение", emotion)
                self.emotions.react_to_emotion(emotion)

                # Анализ аудио и обработка команд
                audio_text = self.environment.capture_audio()
                if audio_text:
                    self.process_voice_command(audio_text)
                    sentiment = self.emotions.analyze_sentiment(audio_text)
                    self.memory.add_experience({
                        "text": audio_text,
                        "sentiment": sentiment
                    })

                # Анализ ошибок и самокоррекция
                self.correction.analyze_errors()

                # Планирование
                self.planning.evaluate_goals()
                self.planning.generate_goals(self.memory.memory["facts"])

                # Анализ причинно-следственных связей
                self.consciousness.analyze_causality()

                # Сохранение состояния нейронной сети
                self.neural_network.save_state()

                # Обработка текста с экрана
                self.process_screen_text()

                # Пауза между действиями
                time.sleep(5)

            except KeyboardInterrupt:
                self.emotions.engine.say("Остановка системы...")
                self.neural_network.stop_network()
                break
            except Exception as e:
                self.correction.add_error(str(e))
                self.emotions.engine.say("Произошла ошибка. Продолжаю работу.")

    def read_book(self, book_path):
        """Чтение книги и извлечение знаний"""
        try:
            if book_path.endswith('.txt'):
                with open(book_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.process_text(text)
                    self.learning_materials["книги"].append({
                        "путь": book_path,
                        "содержание": text,
                        "дата_чтения": time.time()
                    })
                    return True
            elif book_path.endswith('.pdf'):
                with open(book_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                        self.reading_stats["прочитано_страниц"] += 1
                    self.process_text(text)
                    self.learning_materials["pdf"].append({
                        "путь": book_path,
                        "содержание": text,
                        "дата_чтения": time.time()
                    })
                    return True
            elif book_path.endswith('.docx'):
                doc = docx.Document(book_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                self.process_text(text)
                self.learning_materials["docx"].append({
                    "путь": book_path,
                    "содержание": text,
                    "дата_чтения": time.time()
                })
                return True
            return False
        except Exception as e:
            self.correction.add_error(f"Ошибка чтения книги: {e}")
            return False

    def process_text(self, text):
        """Обработка текста и извлечение знаний"""
        # Разбиваем текст на предложения
        sentences = sent_tokenize(text)
        
        # Получаем стоп-слова
        stop_words = set(stopwords.words('russian'))
        
        for sentence in sentences:
            # Анализ предложения
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word not in stop_words]
            
            # Анализ грамматики
            pos_tags = nltk.pos_tag(words)
            
            # Извлечение знаний
            if "правило" in sentence.lower() or "закон" in sentence.lower():
                self.knowledge_base["правила"].append({
                    "текст": sentence,
                    "слова": words,
                    "грамматика": pos_tags
                })
                self.reading_stats["изучено_правил"] += 1
            elif "навык" in sentence.lower() or "умение" in sentence.lower():
                self.knowledge_base["навыки"].append({
                    "текст": sentence,
                    "слова": words,
                    "грамматика": pos_tags
                })
            elif "понятие" in sentence.lower() or "термин" in sentence.lower():
                self.knowledge_base["понятия"].append({
                    "текст": sentence,
                    "слова": words,
                    "грамматика": pos_tags
                })
                self.reading_stats["изучено_понятий"] += 1
            else:
                self.knowledge_base["язык"].append({
                    "текст": sentence,
                    "слова": words,
                    "грамматика": pos_tags
                })
                self.reading_stats["изучено_слов"] += len(words)

    def learn_from_text(self, text):
        """Обучение на основе текста"""
        # Анализ текста
        self.process_text(text)
        
        # Создание новых нейронов для новых знаний
        if len(self.knowledge_base["язык"]) > 0:
            new_neuron = self.neural_network.create_random_neuron()
            if new_neuron:
                new_neuron.name = f"ЗНАНИЕ_{len(self.knowledge_base['язык'])}"
                # Связываем с активными нейронами
                active_neurons = [name for name, neuron in self.neural_network.neurons.items() 
                                if neuron.current_state[0] + neuron.current_state[1] > neuron.treshold 
                                and name != "СЕНСОРНЫЙ"]
                for active_name in active_neurons:
                    self.neural_network.link_neurons(active_name, new_neuron.name)

    def get_reading_stats(self):
        """Получение статистики чтения"""
        return {
            "прочитано_страниц": self.reading_stats["прочитано_страниц"],
            "изучено_слов": self.reading_stats["изучено_слов"],
            "изучено_правил": self.reading_stats["изучено_правил"],
            "изучено_понятий": self.reading_stats["изучено_понятий"]
        }

# ======================
# 13. Запуск системы
# ======================
if __name__ == "__main__":
    ai = AdvancedAI()
    ai.run() 