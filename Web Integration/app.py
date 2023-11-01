from flask import Flask, render_template, request
import os
from scraping import scrape_article
import googletrans
from googletrans import Translator
import time
from transformers import MarianMTModel, MarianTokenizer, T5Tokenizer, T5ForConditionalGeneration
import wikipedia
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import json
import torch
import nltk
import pandas as pd


app = Flask(__name__)

languages = {
    'en': 'English',
    'fr': 'French',
    'de': 'German',
    'es': 'Spanish',
    'ar': 'Arabic',
    'uk': 'Ukrainian',
    'it': 'Italian',
    'ro': 'romanian'
}


@app.route('/')
def index():
    return render_template('index.html', languages=languages)

@app.route('/scrape', methods=['POST'])
def scrape():
    article_input = request.form.get('article_input')
    language = request.form.get('language', 'en')

    start_time = time.time()

    scraped_content = scrape_article(article_input, language)
    if scraped_content:
        word_count = len(scraped_content.split())
        paragraph_count = scraped_content.count('\n\n') + 1
        article_length = len(scraped_content)
        execution_time = calculate_execution_time(start_time)

        return {
            'article': article_input,
            'content': scraped_content,
            'word_count': word_count,
            'paragraph_count': paragraph_count,
            'length_article': article_length,
            'execution_time': execution_time,
            'language': languages.get(language, 'Unknown')
        }
    else:
        return 'Wikipedia page not found. Please try with another title.'

tokenizer = None
model = None

@app.route('/translate', methods=['POST'])
def translate():
    source_article = "\n\n".join(request.form.getlist('source_article'))
    source_language = request.form.get('source_language')  
    destination_language = request.form.get('destination_language')  
    translation_model = request.form.get('translation_model', 'googletranslate')

    native_source_language = languages.get(source_language)
    native_destination_language = languages.get(destination_language)
    print('native_destination_language', native_destination_language)
    print('destination_language', destination_language)

    start_time = time.time()

    translated_content = ""

    if translation_model == "Marian":
        model_name = f'Helsinki-NLP/opus-mt-{source_language}-{destination_language}'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        article_chunks = [source_article[i:i + 500] for i in range(0, len(source_article), 500)]

        translated_chunks = []
        for chunk in article_chunks:
            inputs = tokenizer(chunk, padding=True, truncation=True, return_tensors="pt")
            inputs = inputs.to(device)
            translations = model.generate(**inputs)
            translated_chunk = tokenizer.decode(translations[0], skip_special_tokens=True)
            translated_chunks.append(translated_chunk)

        translated_content = ''.join(translated_chunks)
    
    elif translation_model == "T5":
        if tokenizer is None or model is None:
            tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
            model = T5ForConditionalGeneration.from_pretrained('t5-small')
        words = source_article.split()
        chunks = [words[i:i+40] for i in range(0, len(words), 40)]
    
        translated_content = ""
        for chunk in chunks:
            chunk_text = " ".join(chunk)
            input_ids = tokenizer.encode(f"translate {native_source_language} to {native_destination_language}: "+chunk_text, return_tensors="pt", max_new_tokens=512, truncation=True)
            outputs = model.generate(input_ids, max_new_tokens=100)
            translated_content += tokenizer.decode(outputs[0], skip_special_tokens=True) + " "
        translated_content = ''.join(translated_content)
        
    else:
        translator = Translator()
        for section in source_article.split("\n\n"):
            try:
                translated_section = translator.translate(section, src=source_language, dest=destination_language)
                if translated_section.text:
                    translated_content += translated_section.text + "\n\n"
            except Exception as e:
                print(f"An error occurred during translation: {str(e)}")

    word_count = len(translated_content.split())
    paragraph_count = translated_content.count('\n\n') + 1
    article_length = len(translated_content)
    execution_time = calculate_execution_time(start_time) 

    translated_statistics = {
        'translated_text': translated_content,
        'word_count': word_count,
        'paragraph_count': paragraph_count,
        'length_article': article_length,
        'execution_time': execution_time
    }

    return translated_statistics

@app.route('/compare', methods=['POST'])
def compare():
    destination_language = request.form.get('destination_language', 'fr') 
    translated_article = request.form.get('translated_article') 
    article_title = request.form.get('article_title')
    print("article_title", article_title)
    print("destination_language", destination_language)

    article_output = scrape_article(article_title, destination_language)

    translated_phrases = nltk.sent_tokenize(translated_article)
    article_phrases = nltk.sent_tokenize(article_output)

    df = pd.DataFrame(columns=['translated phrases', 'article phrases'])

    max_length = max(len(translated_phrases), len(article_phrases))
    for i in range(max_length):
        translated_phrase = translated_phrases[i] if i < len(translated_phrases) else ""
        article_phrase = article_phrases[i] if i < len(article_phrases) else ""
        df.loc[i] = [translated_phrase, article_phrase]

    sentences = df['article phrases'].dropna()
    translations = df['translated phrases'].dropna()
    
    sentences=sentences.tolist()
    translations=translations.tolist()

    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model = BertModel.from_pretrained(model_name)

    def get_bert_embeddings(input_texts):
        encoded_inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = bert_model(**encoded_inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.numpy()
    
    original_embeddings = get_bert_embeddings(sentences)
    translated_embeddings = get_bert_embeddings(translations)

    cos_sim = cosine_similarity(translated_embeddings, original_embeddings)

    grouped_data = []
    matched_article_phrases = set()

    for i, translation in enumerate(translations):
        max_similarity = -1
        max_similarity_index = -1

        for j, article_phrase in enumerate(article_phrases):
            if j not in matched_article_phrases:  # Check if the article phrase has been matched before
                similarity_score = cos_sim[i][j]

                if similarity_score > max_similarity:
                    max_similarity = similarity_score
                    max_similarity_index = j

        if max_similarity_index != -1:
            article_phrase = article_phrases[max_similarity_index]
            similarity_score = cos_sim[i][max_similarity_index]

            grouped_data.append({'Index': i, 'Article Phrase': article_phrase, 'Translation': translation, 'Similarity': similarity_score})
            matched_article_phrases.add(max_similarity_index)  # Add the matched article phrase index to the set

    grouped_df = pd.DataFrame(grouped_data)
    grouped_df.set_index('Index', inplace=True)
    grouped_df = grouped_df[(grouped_df['Translation'] != "")]


    print('grouped_df', grouped_df)

    # Recommendation
    missing_parts = ""

    for index, row in grouped_df.iterrows():
        if (row['Similarity'] < 0.9) :
            missing_parts += row['Translation'] + "\n"
    print('missing_parts', missing_parts)

    response = {'missing_parts': missing_parts, 'grouped_df': grouped_df.to_dict()}
    return json.dumps(response)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file and allowed_file(file.filename):
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        return {'content': file_content}, 200

def allowed_file(filename):
    allowed_extensions = {'txt', 'pdf', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def calculate_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int((execution_time % 60))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

if __name__ == '__main__':
    app.run(debug=True)