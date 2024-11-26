import nlpaug.augmenter.word as naw
import pandas as pd

# Загружаем оба файла
processed_data = pd.read_csv("emotions.csv")
augmented_data = pd.read_csv("augmented_emotions.csv")

# Объединяем данные
data = pd.concat([processed_data, augmented_data], ignore_index=True)

# Создаем аугментатор для замены синонимов
aug = naw.SynonymAug(aug_p=0.2)  # aug_p - вероятность замены синонимов

# Первая аугментация
augmented_texts_1 = [aug.augment(text) for text in data['Текст фразы']]

# Обновляем данные с первой аугментацией
data['Текст фразы'] = augmented_texts_1

# Вторая аугментация на уже аугментированных данных
augmented_texts_2 = [aug.augment(text) for text in data['Текст фразы']]

# Обновляем данные с второй аугментацией
data['Текст фразы'] = augmented_texts_2

# Сохраняем обновленные данные в новый файл
data.to_csv("augmented_data.csv", index=False)

print("Двойная аугментация завершена. Данные сохранены в 'augmented_data.csv'.")
