from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim
import numpy as np
from gensim.models import Word2Vec
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
import json

!wget faq.json https://raw.githubusercontent.com/vifirsanova/compling/main/tasks/task3/faq.json
with open('faq.json', encoding='utf-8') as f:
  data = json.load(f)

dp = Dispatcher()
bot = Bot(token='')

faq_questions = []
faq_answers = []
for i in data.values():
    for y in i:
        faq_questions.append(y['question'])
        faq_answers.append(y['answer'])

# TF-IDF преобразование
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions)

# Поиск по лучшему совпадению: Tf-IDF
def best_tfidf_answer(question):
    # Преобразуем запрос в вектор
    query = vectorizer.transform([question])
    similarities = cosine_similarity(query, tfidf_matrix) 
    best_match_idx = similarities.argmax()
    best_answer = faq_answers[best_match_idx]
    return best_answer

# Подгружаем Word2Vec
sentences = [q.split() for q in faq_questions]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Функция для усреднения векторов слов в вопросе
def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) # Берем среднее значение по всем векторам, чтобы одно предложение представлял один вектор

def best_word2wec_answer(question):
    # Векторизуем вопросы
    faq_vectors = np.array([sentence_vector(q, model) for q in faq_questions])
    query_vector = sentence_vector(question, model).reshape(1, -1)
    # Оценка косинусного сходства
    similarities = cosine_similarity(query_vector, faq_vectors)
    best_match_idx = similarities.argmax()
    best_answer = faq_answers[best_match_idx]
    return best_answer

# Создание команды для приветствия
@dp.message (Command('start'))
async def start_message(message: types.Message):
    keyboard = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='О компании')], [KeyboardButton(text='Пожаловаться')]], resize_keyboard=True)
    await message.answer('Привет! Я здесь для того, чтобы ответить на Ваши вопросы. Что Вас интересует?', reply_markup=keyboard)

# Обработка нажатия кнопки "О компании"
@dp.message (lambda message: message.text == 'О компании')
async def about_company(message: types.Message):
    await message.answer('Наша компания занимается доставкой товаров по всей стране.')

# Обработка нажатия кнопки "Пожаловаться"
@dp.message (lambda message: message.text == 'Пожаловаться')
async def to_complain(message: types.Message):
    await message.answer('Отправьте скриншот для того, чтоб я передал запрос с жалобой специалисту')

# Обработка изображений
@dp.message (lambda message: message.content_type == 'photo')
async def photo(message: types.Message):
    file_id = message.photo[-1].file_id
    file = await bot.get_file(file_id)
    filename = file.file_path.split("/")[-1]
    filesize = message.photo[0].file_size
    await message.answer(f'Название файла: {filename}, размер: {filesize} байт. Передаю Ваш запрос специалисту.')

# Обработка вопросов
@dp.message()
async def answer(message: types.Message):
    question = message.text
    answer_1 = best_tfidf_answer(question)
    answer_2 = best_word2wec_answer(question)
    await message.answer(f"Ответ 1: {answer_1}")
    await message.answer(f"Ответ 2: {answer_2}")

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    await main()
