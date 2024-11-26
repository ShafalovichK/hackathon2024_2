import concurrent.futures
import nlpaug.augmenter.word as naw
import pandas as pd
import tqdm  # Для прогресс-бара

def augment_single_text(text, back_translation_aug):
    try:
        # Агментация текста с использованием перевода
        return back_translation_aug.augment(text)
    except Exception as e:
        print(f"Error augmenting text: {text[:30]}... Error: {e}")
        return text  # Возвращаем исходный текст, если произошла ошибка

def augment_data(data, emotion, n_syn=10, n_back=5, chunk_size=100):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='Helsinki-NLP/opus-mt-en-ru', 
        to_model_name='Helsinki-NLP/opus-mt-ru-en'
    )
    
    augmented_data = []
    
    # Обрабатываем данные по частям (например, по 100 строк)
    for i in tqdm.tqdm(range(0, len(data), chunk_size), desc="Processing data in chunks"):
        chunk = data.iloc[i:i+chunk_size]
        # Используем ThreadPoolExecutor для параллельной обработки
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for augmented_text in executor.map(lambda text: augment_single_text(text, back_translation_aug), chunk['Текст фразы']):
                augmented_data.append(augmented_text)
    
    data['Augmented'] = augmented_data
    return data

if __name__ == '__main__':
    try:
        # Загружаем обработанные данные
        data = pd.read_csv(r"processed_emotions.csv")
        print(f"Data loaded successfully. First few rows:\n{data.head()}")

        # Применяем агментацию
        new_data = augment_data(data, 'Недовольство', n_syn=10, n_back=5)
        
        # Выводим результат
        print(new_data.head())
        
        # Сохраняем результаты в новый файл
        new_data.to_csv(r"augmented_emotions.csv", index=False)
    
    except Exception as e:
        print(f"Error: {e}")

