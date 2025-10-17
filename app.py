import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import plotly.graph_objects as go

# ==================== KONFIGURASI ====================
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🖼️",
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
    """Load trained model dan class labels"""
    try:
        model = tf.keras.models.load_model('optimized_cnn_model.h5')
        
        with open('class_labels.json', 'r') as f:
            class_labels = json.load(f)
        
        return model, class_labels
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# ==================== PREPROCESSING ====================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image untuk prediksi"""
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(img)
    
    # Ensure RGB
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ==================== PREDIKSI ====================
def predict_image(model, image, class_labels):
    """Melakukan prediksi pada gambar"""
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_img, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Get class name
    predicted_class = class_labels[str(predicted_class_idx)]
    
    # Get all probabilities
    all_probs = {class_labels[str(i)]: float(predictions[0][i]) 
                 for i in range(len(class_labels))}
    
    return predicted_class, confidence, all_probs

# ==================== VISUALISASI ====================
def plot_probabilities(probabilities):
    """Plot probability distribution menggunakan Plotly"""
    # Sort probabilities
    sorted_probs = dict(sorted(probabilities.items(), 
                               key=lambda x: x[1], 
                               reverse=True))
    
    classes = list(sorted_probs.keys())
    probs = list(sorted_probs.values())
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker=dict(
                color=probs,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f'{p:.2%}' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Prediction Probabilities',
        xaxis_title='Probability',
        yaxis_title='Class',
        height=400,
        template='plotly_white'
    )
    
    return fig

# ==================== MAIN APP ====================
def main():
    # Header
    st.title("🖼️ CNN Image Classifier")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        model, class_labels = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if model file exists.")
        return
    
    st.success(f"✅ Model loaded successfully! ({len(class_labels)} classes)")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Informasi")
        st.info(f"""
        **Model:** Optimized CNN dengan Transfer Learning
        
        **Jumlah Kelas:** {len(class_labels)}
        
        **Kelas yang tersedia:**
        {', '.join(class_labels.values())}
        """)
        
        st.markdown("---")
        
        st.header("⚙️ Settings")
        show_probabilities = st.checkbox("Tampilkan Semua Probabilitas", value=True)
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
        st.header("📤 Upload Gambar")
        
        uploaded_file = st.file_uploader(
            "Pilih gambar...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload gambar dalam format JPG, JPEG, PNG, atau BMP"
        )
        
        if uploaded_file is not None:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display original image
            st.image(image, caption='Gambar Original', use_container_width=True)
            
            # Predict button
            if st.button("🔍 Prediksi", key="predict_btn"):
                with st.spinner("Memproses prediksi..."):
                    # Predict
                    predicted_class, confidence, all_probs = predict_image(
                        model, image, class_labels
                    )
                    
                    # Store in session state
                    st.session_state['predicted_class'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['all_probs'] = all_probs
    
    with col2:
        st.header("📊 Hasil Prediksi")
        
        if 'predicted_class' in st.session_state:
            predicted_class = st.session_state['predicted_class']
            confidence = st.session_state['confidence']
            all_probs = st.session_state['all_probs']
            
            # Display result
            if confidence >= confidence_threshold:
                st.success(f"### ✅ Prediksi: **{predicted_class}**")
            else:
                st.warning(f"### ⚠️ Prediksi: **{predicted_class}** (Low Confidence)")
            
            # Confidence meter
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2%}",
                delta=f"{confidence - confidence_threshold:.2%} dari threshold"
            )
            
            # Progress bar
            st.progress(float(confidence))
            
            # Show all probabilities
            if show_probabilities:
                st.markdown("---")
                st.subheader("📈 Distribution Probabilitas")
                fig = plot_probabilities(all_probs)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                with st.expander("📋 Lihat Detail Probabilitas"):
                    sorted_probs = dict(sorted(all_probs.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
                    for cls, prob in sorted_probs.items():
                        st.write(f"**{cls}:** {prob:.4f} ({prob*100:.2f}%)")
        else:
            st.info("👆 Upload gambar dan klik tombol Prediksi untuk melihat hasil")
            
            # Example images
            st.markdown("---")
            st.subheader("💡 Tips:")
            st.markdown("""
            - Gunakan gambar dengan kualitas baik
            - Pastikan objek terlihat jelas
            - Hindari gambar yang terlalu gelap atau blur
            - Ukuran file maksimal 200MB
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚀 Powered by TensorFlow & Streamlit | Made with ❤️</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()