from transformers import TFXLMRobertaModel, XLMRobertaTokenizer
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import pickle
import numpy as np

# Загрузка данных
raw_data_path = "emotions.csv"
augmented_data_path = "augmented_data.csv"

raw_data = pd.read_csv(raw_data_path)
augmented_data = pd.read_csv(augmented_data_path)

# Преобразование меток эмоций в числовой формат
raw_data['Эмоция'] = raw_data['Эмоция'].astype('category').cat.codes
augmented_data['Эмоция'] = augmented_data['Эмоция'].astype('category').cat.codes

# Инициализация токенизатора
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# Разделение сырых данных на обучающую и тестовую выборки
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    raw_data['Текст фразы'], raw_data['Эмоция'], test_size=0.2, random_state=42
)

# Объединение сырых и аугментированных данных для второго этапа
combined_data = pd.concat([raw_data, augmented_data], ignore_index=True)
X_train_combined = combined_data['Текст фразы']
y_train_combined = combined_data['Эмоция']

# Токенизация данных
def tokenize_texts(texts):
    return tokenizer(
        texts.tolist(),
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Загрузка модели XLM-RoBERTa для получения эмбеддингов
embedding_model = TFXLMRobertaModel.from_pretrained('xlm-roberta-base')

# Заморозка первых шести слоёв модели
for layer in embedding_model.roberta.encoder.layer[:6]:
    layer.trainable = False

# Оптимизация с использованием шедулера
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
embedding_model.compile(optimizer=optimizer, loss='categorical_crossentropy')

# Получение эмбеддингов батчами
def get_embeddings_batchwise(texts, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenize_texts(batch_texts)
        batch_embeddings = embedding_model(tokens)['last_hidden_state'][:, 0, :].numpy()  # Первый токен [CLS]
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Главная функция
if __name__ == "__main__":
    # Первый этап: обучение на сырых данных
    print("Этап 1: обучение на сырых данных...")
    train_embeddings_raw = get_embeddings_batchwise(X_train_raw)
    test_embeddings = get_embeddings_batchwise(X_test)

    svm = SVC(kernel='linear', C=1, probability=True)
    svm.fit(train_embeddings_raw, y_train_raw)

    # Оценка на тестовой выборке после первого этапа
    svm_preds_raw = svm.predict(test_embeddings)
    accuracy_raw = accuracy_score(y_test, svm_preds_raw)
    print(f"Точность на сырых данных: {accuracy_raw:.4f}")

    # Второй этап: дообучение на объединенных данных
    print("Этап 2: дообучение на объединенных данных...")
    train_embeddings_combined = get_embeddings_batchwise(X_train_combined)
    svm.fit(train_embeddings_combined, y_train_combined)

    # Оценка на тестовой выборке после второго этапа
    svm_preds_combined = svm.predict(test_embeddings)
    accuracy_combined = accuracy_score(y_test, svm_preds_combined)
    precision_combined = precision_score(y_test, svm_preds_combined, average='weighted')
    recall_combined = recall_score(y_test, svm_preds_combined, average='weighted')
    f1_combined = f1_score(y_test, svm_preds_combined, average='weighted')

    # Вывод метрик
    print(f"Точность на объединенных данных: {accuracy_combined:.4f}")
    print(f"Precision: {precision_combined:.4f}")
    print(f"Recall: {recall_combined:.4f}")
    print(f"F1-Score: {f1_combined:.4f}")

    # Сохранение модели SVM
    model_path = "svm_emotion_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(svm, f)
    print(f"Модель SVM сохранена в {model_path}")





