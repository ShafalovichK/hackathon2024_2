import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import pymorphy2

# Инициализация лемматизатора
morph = pymorphy2.MorphAnalyzer()

# Загрузка данных
data = pd.read_csv('emotions.csv')

# Загрузка стоп-слов
nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

# Функция для предобработки текста
def preprocess_text(text):
    text = text.lower()  # Нормализация к нижнему регистру
    text = re.sub(r'\d+', '', text)  # Удаление цифр
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    words = [word for word in text.split() if word not in stop_words]  # Удаление стоп-слов
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]  # Лемматизация
    return ' '.join(lemmatized_words)

# Применение предобработки
data['Текст фразы'] = data['Текст фразы'].apply(preprocess_text)

# Проверка результата
print(data.head())

# Сохранение обработанных данных
data.to_csv('processed_emotions.csv', index=False)

