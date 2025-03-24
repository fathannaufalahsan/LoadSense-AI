import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from fpdf import FPDF
import shap
import os
import logging
import streamlit as st
import pygame
import time

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Streamlit Page Configuration
st.set_page_config(page_title="AI-Structural Load Predictor", layout="wide", page_icon="ahsankarya.ico")
st.sidebar.image("ahsantech.png", use_container_width=True)
st.sidebar.markdown("---")
st.sidebar.title("🤖 AI-Structural Load Predictor")
st.sidebar.write("### Information")

# Pilihan Bahasa
language = st.sidebar.radio("🌍 Select Language / Pilih Bahasa", ("English", "Bahasa Indonesia"))

# Deskripsi berdasarkan bahasa yang dipilih
if language == "English":
    description = """
    <div style="text-align: justify;">
    ✨ <b>AI Structural Load Predictor</b> is an advanced <i>machine learning</i>-based application designed to analyze and predict the maximum load a structure can withstand. By considering <b>material characteristics, dimensions, and environmental conditions</b>, this application helps engineers and researchers in designing and analyzing <b>safer and more efficient structures</b>.  

    🧠 Built with <i>Python</i> and powered by <i>Neural Networks (Deep Learning)</i>, this application has been trained using structural datasets to provide <b>fast, accurate, and data-driven predictions</b>. Combining <b>cutting-edge AI</b> with <b>interactive visualization</b>, <b>AI Structural Load Predictor</b> introduces a new era of innovation in civil engineering and structural analysis.
    </div>
    """
else:
    description = """
    <div style="text-align: justify;">
    ✨ <b>AI Structural Load Predictor</b> adalah aplikasi canggih berbasis <i>machine learning</i> yang dirancang untuk menganalisis dan memprediksi beban maksimum yang dapat ditahan oleh suatu struktur. Dengan mempertimbangkan <b>karakteristik material, dimensi, dan kondisi lingkungan</b>, aplikasi ini membantu insinyur dan peneliti dalam perancangan dan analisis struktur yang lebih <b>aman dan efisien</b>.  

    🧠 Dibangun dengan <i>Python</i> dan didukung oleh <i>Neural Network (Deep Learning)</i>, aplikasi ini telah dilatih menggunakan dataset struktural untuk memberikan <b>prediksi yang cepat, akurat, dan berbasis data nyata</b>. Dengan kombinasi <b>AI mutakhir</b> dan <b>visualisasi interaktif</b>, <b>AI Structural Load Predictor</b> menghadirkan inovasi baru dalam dunia teknik sipil dan rekayasa struktur.
    </div>
    """

# Menampilkan deskripsi di sidebar
st.sidebar.markdown(description, unsafe_allow_html=True)

# Developer Information
st.sidebar.markdown("----")
st.sidebar.write("### 👨‍💻 Developer Information")
st.sidebar.write("**Name:** Fathan Naufal Ahsan")
st.sidebar.write("**Brand:** Ahsan Karya")
st.sidebar.write("**Email:** [fathannaufalahsan.18@gmail.com](mailto:fathannaufalahsan.18@gmail.com)")

# Pemisah
st.sidebar.markdown("----")

# Pilihan Bahasa dengan Unique Key
language = st.sidebar.radio("🌍 Select Language / Pilih Bahasa", ("English", "Bahasa Indonesia"), key="language_selector")

# Noted Section berdasarkan bahasa yang dipilih
if language == "English":
    note_text = """
    **📝 Important Notes:**  
    1️⃣ If a **server error** occurs, please **refresh the page**. 🔄  
    2️⃣ The process may take **few seconds** as the system processes **dataset parameters**. Please be patient. ⏳  
    """
else:
    note_text = """
    **📝 Catatan Penting:**  
    1️⃣ Jika terjadi **error pada server**, silakan **muat ulang halaman**. 🔄  
    2️⃣ Proses memerlukan waktu **beberapa detik** karena sistem memproses **parameter dataset**. Harap bersabar. ⏳  
    """
# Menampilkan Noted Section
st.sidebar.markdown(note_text)

# Pemisah
st.sidebar.markdown("----")

# Sidebar Controls for Model Training
st.sidebar.write("### ⚙️ Model Parameters")
epochs = st.sidebar.slider("Epochs", 10, 500, 100, step=10)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32, step=16)

# Load dataset (Dummy dataset, replace with real data in production)
@st.cache_data
def generate_dummy_data():
    """
    Generate a dummy dataset for model training and testing.

    Returns:
        pd.DataFrame: A DataFrame containing structural parameters and a target variable (max_load).
    """
    data = pd.DataFrame({
        'material_strength': np.random.uniform(100, 500, 1000),
        'elastic_modulus': np.random.uniform(20000, 200000, 1000),
        'height': np.random.uniform(0.1, 2, 1000),
        'width': np.random.uniform(0.1, 2, 1000),
        'thickness': np.random.uniform(0.01, 0.2, 1000),
        'temperature': np.random.uniform(-10, 50, 1000),
        'humidity': np.random.uniform(10, 90, 1000),
        'max_load': np.random.uniform(1000, 10000, 1000)
    })
    return data

data = generate_dummy_data()

# Display dataset visualizations in the sidebar
if st.sidebar.checkbox("📊 Explore Dataset"):
    st.sidebar.write("### Dataset Summary")
    st.sidebar.write(data.describe())
    st.sidebar.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.sidebar.pyplot(fig)

    st.sidebar.write("### Data Distribution")
    column = st.sidebar.selectbox("Select Column", data.columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data[column], kde=True, ax=ax)
    st.sidebar.pyplot(fig)

# Split dataset into training and testing sets
X = data.drop(columns=['max_load'])
y = data['max_load']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and Save AI Model if not exists
model_path = "ai_load_predictor.keras"
if not os.path.exists(model_path):
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compile Model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train Model
    logging.info("Training model...")
    with st.spinner("Training the model... This may take a while."):
        history = model.fit(X_train_scaled, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test_scaled, y_test), verbose=0)

    # Save Model
    model.save(model_path)
    logging.info("Model saved.")

# Load Model
model = keras.models.load_model(model_path)

# SHAP Explainer
explainer = shap.Explainer(model, X_test_scaled)
shap_values = explainer(X_test_scaled)


# Prediction Function
def predict_load(material_strength, elastic_modulus, height, width, thickness, temperature, humidity):
    """
    Predict the maximum load based on input parameters.

    Args:
        material_strength (float): Material strength in MPa.
        elastic_modulus (float): Elastic modulus in MPa.
        height (float): Height in meters.
        width (float): Width in meters.
        thickness (float): Thickness in meters.
        temperature (float): Temperature in °C.
        humidity (float): Humidity as a percentage.

    Returns:
        float: Predicted maximum load in Newtons.
    """
    try:
        input_data = np.array([[material_strength, elastic_modulus, height, width, thickness, temperature, humidity]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        return prediction[0][0]
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return None


# Generate PDF Report
def generate_pdf(material_strength, elastic_modulus, height, width, thickness, temperature, humidity, prediction):
    """
    Generate a PDF report of the prediction results.

    Args:
        material_strength (float): Material strength in MPa.
        elastic_modulus (float): Elastic modulus in MPa.
        height (float): Height in meters.
        width (float): Width in meters.
        thickness (float): Thickness in meters.
        temperature (float): Temperature in °C.
        humidity (float): Humidity as a percentage.
        prediction (float): Predicted maximum load in Newtons.

    Returns:
        None
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "AI-Structural Load Prediction Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, "Developer: Fathan Naufal Ahsan", ln=True)
    pdf.cell(200, 10, "Brand: Ahsan Karya", ln=True)
    pdf.cell(200, 10, "Email: fathannaufalahsan.18@gmail.com", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, f"Material Strength: {material_strength} MPa", ln=True)
    pdf.cell(200, 10, f"Elastic Modulus: {elastic_modulus} MPa", ln=True)
    pdf.cell(200, 10, f"Height: {height} m", ln=True)
    pdf.cell(200, 10, f"Width: {width} m", ln=True)
    pdf.cell(200, 10, f"Thickness: {thickness} m", ln=True)
    pdf.cell(200, 10, f"Temperature: {temperature} °C", ln=True)
    pdf.cell(200, 10, f"Humidity: {humidity} %", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, f"Predicted Maximum Load: {prediction:.2f} N", ln=True)
    pdf.output("prediction_report.pdf")

# Initialize pygame mixer
pygame.mixer.init()
def play_welcome_audio():
    try:
        pygame.mixer.music.load("robot_voice.wav")
        pygame.mixer.music.play()
        time.sleep(3)  # Tunggu agar suara selesai diputar
    except Exception as e:
        st.error(f"Gagal memutar audio: {e}")

# Main UI
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Poppins:wght@300;400;600&display=swap');

        /* Warna latar belakang */
        body {
            background-color: #0d1117;
            color: #ffffff;
        }

        /* Animasi Fade-in */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Animasi Glow */
        @keyframes glow {
            0% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
            50% { text-shadow: 0 0 10px #00e6e6, 0 0 20px #00e6e6, 0 0 30px #00e6e6; }
            100% { text-shadow: 0 0 5px #33ccff, 0 0 10px #33ccff, 0 0 15px #33ccff; }
        }

        /* Judul */
        .title {
            text-align: center;
            color: #33ccff;
            font-size: 3rem;
            font-family: 'Orbitron', sans-serif;
            font-weight: 700;
            animation: fadeIn 2s ease-in-out, glow 3s infinite alternate;
        }

        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #00e6e6;
            font-size: 1.2rem;
            font-family: 'Poppins', sans-serif;
            font-weight: 400;
            animation: fadeIn 3s ease-in-out;
        }

        /* Garis pemisah */
        .separator {
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, #33ccff, #00e6e6);
            margin: 20px auto;
            animation: fadeIn 1.5s ease-in-out;
        }

        /* Tombol interaktif */
        .ai-button {
            display: block;
            width: 250px;
            margin: 30px auto;
            padding: 15px;
            text-align: center;
            font-size: 1rem;
            font-family: 'Poppins', sans-serif;
            color: #fff;
            background: linear-gradient(90deg, #33ccff, #00e6e6);
            border: none;
            border-radius: 25px;
            cursor: pointer;
            transition: 0.3s ease-in-out;
            box-shadow: 0 0 10px #00e6e6;
        }

        .ai-button:hover {
            background: linear-gradient(90deg, #00e6e6, #33ccff);
            box-shadow: 0 0 20px #00e6e6;
        }
    </style>

    <h1 class="title">🤖 AI-Structural Load Predictor</h1>
    <h4 class="subtitle">Predict the maximum load capacity with AI technology</h4>
    <div class="separator"></div>
    """,
    unsafe_allow_html=True
)

# Play welcome audio when the app starts
if 'audio_played' not in st.session_state:
    st.session_state.audio_played = False

if not st.session_state.audio_played:
    play_welcome_audio()
    st.session_state.audio_played = True

# Layout Columns
with st.container():
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        material_strength = st.number_input("🔩 Material Strength (MPa)", 100, 500, 250)
        elastic_modulus = st.number_input("📏 Elastic Modulus (MPa)", 20000, 200000, 100000)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        height = st.number_input("📐 Height (m)", 0.1, 2.0, 1.0)
        width = st.number_input("📏 Width (m)", 0.1, 2.0, 1.0)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        thickness = st.number_input("📏 Thickness (m)", 0.01, 0.2, 0.1)
        temperature = st.number_input("🌡️ Temperature (°C)", -10, 50, 20)
        humidity = st.number_input("💧 Humidity (%)", 10, 90, 50)
        st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Predict Load", help="Click to predict the structural load capacity"):
    with st.spinner("🔄 Processing... Please wait!"):
        st.markdown('<div class="loading-icon"></div>', unsafe_allow_html=True)
        prediction = predict_load(material_strength, elastic_modulus, height, width, thickness, temperature, humidity)

    if prediction is not None:
        st.success(f"✅ Predicted Maximum Load: **{prediction:.2f} N**")

        # SHAP Feature Contribution
        st.write("### 🔍 Feature Contribution to Prediction")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

        # Generate and Download PDF Report
        generate_pdf(material_strength, elastic_modulus, height, width, thickness, temperature, humidity, prediction)
        with open("prediction_report.pdf", "rb") as file:
            st.download_button(
                label="📄 Download Prediction Report",
                data=file,
                file_name="prediction_report.pdf",
                mime="application/pdf"
            )
    else:
        st.error("❌ Prediction failed. Please check the inputs or try again.")
