import pandas as pd
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Unduh data NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Membaca file CSV
data = pd.read_csv('tempat_makan.csv', delimiter=';', encoding='latin1')

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Membuat deskripsi tempat makan dalam bentuk list
tempat_makan_deskripsi = data['desk'].tolist()

# Keyword
GREETING_INPUTS = ("halo", "hi", "hai")
GREETING_RESPONSES = ["Halo! Ada yang bisa saya bantu?", "Hi! Butuh rekomendasi tempat makan?", "Halo, selamat datang!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def build_vsm_responses(user_response):
    """Menggunakan Vector Space Model untuk mencari kecocokan."""
    # Inisialisasi respon chatbot
    robo_response = ''
    
    # Menambahkan input pengguna ke deskripsi tempat makan untuk diproses
    tempat_makan_deskripsi.append(user_response)

    # Membuat representasi vektor TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_matrix = TfidfVec.fit_transform(tempat_makan_deskripsi)

    # Menghitung kesamaan kosinus
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    similarity_scores = cosine_similarities.flatten()
    sorted_indices = similarity_scores.argsort()[::-1][1:]

    # Mendapatkan hasil terbaik
    if similarity_scores[sorted_indices[0]] == 0:
        robo_response = "Maaf, saya tidak menemukan tempat makan yang cocok."
    else:
        best_match_index = sorted_indices[0]
        matched_place = data.iloc[best_match_index]
        robo_response = (
            f"Rekomendasi: {matched_place['tempat']}; "
            f"Alamat: {matched_place['alamat']}; "
            f"Kategori: {matched_place['kategori']}; "
            f"Rating: {matched_place['rating']}; "
            f"Harga: Rp.{matched_place['harga']}; "
            f"Deskripsi: {matched_place['desk']}"
        )

    # Menghapus input pengguna dari list deskripsi untuk menjaga konsistensi data
    tempat_makan_deskripsi.pop()
    return robo_response

ADDITIONAL_RESPONSES = {
    "siapa kamu": "Saya chatbot MakanBuddy, silakan bertanya!",
    "siapa namamu": "Nama saya adalah MakanBuddy, asisten Anda dalam mencari tempat makan.",
    "selamat tinggal": "Selamat tinggal!",
    "terima kasih": "Sama-sama!",
}

def response(user_response):
    user_response = user_response.lower()
    if user_response in ADDITIONAL_RESPONSES:
        return ADDITIONAL_RESPONSES[user_response]
    if greeting(user_response):
        return greeting(user_response)
    return build_vsm_responses(user_response)

