import streamlit as st
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Game Chat Moderator",
    page_icon="🎮",
    layout="centered"
)

# --- CSS TEMA GAMING/DARK MODE ---
st.markdown("""
<style>
    /* Background & Text Color */
    .stApp {
        background-color: #0F172A;
        color: #F1F5F9;
    }
    /* Input Box */
    .stTextInput > div > div > input {
        background-color: #1E293B;
        color: white;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    /* Tombol */
    .stButton > button {
        background-color: #6366F1;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        height: 50px;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #4F46E5;
        box-shadow: 0 0 15px #6366F1;
    }
    /* Hasil Prediksi */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        animation: fadeIn 0.5s;
    }
    .toxic {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
        border: 2px solid #F87171;
        box-shadow: 0 0 20px rgba(248, 113, 113, 0.4);
    }
    .safe {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border: 2px solid #34D399;
        box-shadow: 0 0 20px rgba(52, 211, 153, 0.4);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# --- FUNGSI LOAD ASSETS ---
@st.cache_resource
def load_resources():
    # Cek keberadaan file
    if not os.path.exists('model_toxic_game.h5') or not os.path.exists('tokenizer.pickle'):
        return None, None, None

    try:
        # Load Model
        model = load_model('model_toxic_game.h5')
        # Load Tokenizer
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        # Init Stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return model, tokenizer, stemmer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, tokenizer, stemmer = load_resources()

# --- FUNGSI CLEANING TEXT ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'#\w+','',text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus angka/simbol
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- TAMPILAN UTAMA (UI) ---
st.title("🎮 Auto-Moderator Chat")
st.markdown("Sistem deteksi toksisitas chat game berbasis **Hybrid CNN-LSTM**.")

# Cek status model
if model is None:
    st.error("⚠️ **FILE HILANG!**\nPastikan file `model_toxic_game.h5` dan `tokenizer.pickle` sudah ada di dalam folder project ini.")
else:
    # Form Input
    with st.form("chat_form"):
        user_input = st.text_input("Player Chat:", placeholder="Ketikan pesan di sini (misal: ez game, noob lu)...")
        submitted = st.form_submit_button("🛡️ SCAN PESAN")

    if submitted and user_input:
        # 1. Preprocessing
        clean_input = clean_text(user_input)
        
        # 2. Tokenizing
        seq = tokenizer.texts_to_sequences([clean_input])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
        
        # 3. Prediksi
        prediction = model.predict(padded)
        score = prediction[0][0] # Output 0.0 s/d 1.0
        
        # 4. Tampilkan Hasil
        st.markdown("---")
        
        threshold = 0.5 # Batas ambang
        
        if score > threshold:
            # TOXIC
            confidence = score * 100
            st.markdown(f"""
            <div class="result-box toxic">
                <h1 style='margin:0'>🚫 DIBLOKIR</h1>
                <h3 style='margin:0'>TOXIC DETECTED ({confidence:.1f}%)</h3>
                <hr style='border-color:white; opacity:0.3'>
                <p><i>"{user_input}"</i></p>
                <p><b>Sistem:</b> Pesan mengandung ujaran kebencian/kasar.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # SAFE
            confidence = (1 - score) * 100
            st.markdown(f"""
            <div class="result-box safe">
                <h1 style='margin:0'>✅ DITERIMA</h1>
                <h3 style='margin:0'>SAFE / CLEAN ({confidence:.1f}%)</h3>
                <hr style='border-color:white; opacity:0.3'>
                <p><i>"{user_input}"</i></p>
                <p><b>Sistem:</b> Pesan aman untuk dikirim ke publik.</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br><p style='text-align:center; color:#64748B; font-size:12px'>Skripsi Universitas Dinamika | Hybrid CNN-LSTM + Tomek Links</p>", unsafe_allow_html=True)