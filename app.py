import streamlit as st
import numpy as np
import pickle
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Game Chat Moderator",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS CUSTOM (Hanya untuk Kotak Hasil) ---
# Kita HAPUS bagian .stApp background agar mengikuti tema sistem (Light/Dark)
st.markdown("""
<style>
    /* Styling khusus untuk kotak hasil prediksi agar tetap kontras */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
        color: white; /* Teks di dalam kotak hasil selalu putih */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .toxic {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        border: 2px solid #F87171;
    }
    .safe {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border: 2px solid #34D399;
    }
    /* Agar tombol sedikit lebih menarik tapi tetap netral */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 50px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. FUNGSI LOAD ASSETS ---
@st.cache_resource
def load_resources():
    if not os.path.exists('model_toxic_game.h5') or not os.path.exists('tokenizer.pickle'):
        return None, None, None

    try:
        model = load_model('model_toxic_game.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        return model, tokenizer, stemmer
    except Exception as e:
        return None, None, None

model, tokenizer, stemmer = load_resources()

# --- 4. FUNGSI CLEANING ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+','',text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 5. TAMPILAN UTAMA (UI) ---
st.title("🎮 Auto-Moderator Chat")
st.write("Sistem deteksi komentar toksik pada chat game.")

if model is None:
    st.error("⚠️ File Model Hilang! Pastikan 'model_toxic_game.h5' dan 'tokenizer.pickle' ada.")
else:
    with st.form("chat_form"):
        # Input box akan otomatis mengikuti tema (putih di light mode, gelap di dark mode)
        user_input = st.text_input("Masukkan Chat Player:", placeholder="Contoh: mainnya hebat banget...")
        submitted = st.form_submit_button("🔍 Cek Toksisitas")

    if submitted and user_input:
        # Proses
        clean_input = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([clean_input])
        padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
        
        prediction = model.predict(padded)
        score = prediction[0][0]
        
        st.markdown("---")
        
        # Logika Tampilan
        if score > 0.5:
            confidence = score * 100
            st.markdown(f"""
            <div class="result-box toxic">
                <h2 style='margin:0; color:white;'>🚫 TOXIC</h2>
                <p style='margin:5px 0 0 0; color:#FECACA;'>Confidence: {confidence:.1f}%</p>
                <hr style='border-color:rgba(255,255,255,0.3); margin: 10px 0;'>
                <p style='font-style:italic; font-size:14px; color:white;'>"{user_input}"</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            confidence = (1 - score) * 100
            st.markdown(f"""
            <div class="result-box safe">
                <h2 style='margin:0; color:white;'>✅ AMAN</h2>
                <p style='margin:5px 0 0 0; color:#A7F3D0;'>Confidence: {confidence:.1f}%</p>
                <hr style='border-color:rgba(255,255,255,0.3); margin: 10px 0;'>
                <p style='font-style:italic; font-size:14px; color:white;'>"{user_input}"</p>
            </div>
            """, unsafe_allow_html=True)

# Footer sederhana
st.caption("Project Deep Learning: Hybrid CNN-LSTM + Tomek Links")