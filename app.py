import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import plotly.graph_objects as go

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="Optimized CNN - Group 5",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
    }
    .prediction-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Memuat model terlatih dan label kelas."""
    try:
        model = tf.keras.models.load_model('outputs/optimized_cnn_model.h5')
        with open('outputs/class_labels.json', 'r') as f:
            class_labels = json.load(f)
        return model, class_labels
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {str(e)}")
        return None, None

# ==================== PREPROCESSING ====================
def preprocess_image(image, target_size=(224, 224)):
    # Pastikan RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize & normalisasi
    img = image.resize(target_size, Image.LANCZOS)
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ==================== PREDIKSI ====================
def predict_image(model, image, class_labels):
    """Melakukan prediksi terhadap gambar yang diupload."""
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_labels[str(predicted_class_idx)]

    all_probs = {class_labels[str(i)]: float(predictions[0][i]) 
                 for i in range(len(class_labels))}
    return predicted_class, confidence, all_probs

# ==================== VISUALISASI ====================
def plot_probabilities(probabilities):
    """Menampilkan distribusi probabilitas dalam bentuk grafik batang."""
    sorted_probs = dict(sorted(probabilities.items(), 
                               key=lambda x: x[1], 
                               reverse=True))
    classes = list(sorted_probs.keys())
    probs = list(sorted_probs.values())

    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Probabilitas")
            ),
            text=[f'{p:.2%}' for p in probs],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title='Distribusi Probabilitas Prediksi',
        xaxis_title='Probabilitas',
        yaxis_title='Kelas',
        height=400,
        template='plotly_white'
    )
    return fig

# ==================== MAIN APP ====================
def main():
    st.title("Optimized CNN Image Classifier - Group 5 - Artificial Intelligence")
    st.markdown("---")

    # Load model
    with st.spinner("Memuat model..."):
        model, class_labels = load_model()

    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file model tersedia.")
        return

    st.success("Model berhasil dimuat.")

    # Sidebar
    with st.sidebar:
        st.header("Informasi Model")
        st.info(f"""
        Model: CNN yang dioptimalkan dengan transfer learning  
        Jumlah kelas: {len(class_labels)}  
        Kelas yang tersedia: {', '.join(class_labels.values())}
        """)
        st.markdown("---")

        st.header("Pengaturan")
        show_probabilities = st.checkbox("Tampilkan semua probabilitas", value=True)
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )

    # Main content
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload Gambar Anda")

        uploaded_file = st.file_uploader(
            "Pilih gambar untuk diklasifikasikan:",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Format yang didukung: JPG, JPEG, PNG, BMP."
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diupload', use_container_width=True)

            if st.button("Prediksi", key="predict_btn"):
                with st.spinner("Memproses gambar..."):
                    predicted_class, confidence, all_probs = predict_image(model, image, class_labels)
                    st.session_state['predicted_class'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['all_probs'] = all_probs

    with col2:
        st.header("Hasil Prediksi")

        if 'predicted_class' in st.session_state:
            predicted_class = st.session_state['predicted_class']
            confidence = st.session_state['confidence']
            all_probs = st.session_state['all_probs']

            if confidence >= confidence_threshold:
                st.success(f"Hasil Prediksi: {predicted_class}")
            else:
                st.warning(f"Hasil Prediksi: {predicted_class} (Tingkat kepercayaan rendah)")

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2%}",
                delta=f"{confidence - confidence_threshold:.2%} dari batas minimal"
            )

            st.progress(float(confidence))

            if show_probabilities:
                st.markdown("---")
                st.subheader("Distribusi Probabilitas")
                fig = plot_probabilities(all_probs)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Detail Probabilitas"):
                    sorted_probs = dict(sorted(all_probs.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
                    for cls, prob in sorted_probs.items():
                        st.write(f"{cls}: {prob:.4f} ({prob*100:.2f}%)")
        else:
            st.info("upload gambar dan klik tombol 'Prediksi' untuk melihat hasil.")
            st.markdown("---")
            st.subheader("Tips:")
            st.markdown("""
            - Gunakan gambar dengan kualitas baik dan objek yang jelas terlihat.  
            - Hindari gambar yang terlalu gelap, blur, atau memiliki noise tinggi.  
            - Ukuran file maksimal 200 MB.
            """)

if __name__ == "__main__":
    main()
