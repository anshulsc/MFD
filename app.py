import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Fact-Checker AI",
    page_icon="🤖",
    layout="wide"
)

BACKEND_URL = "http://127.0.0.1:8000/predict/"


st.markdown("""
<style>
    /* Center the title */
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        text-align: center;
        color: #FFFFFF; /* White text */
    }
    /* Style for the result label */
    .result-label {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px; /* Add space below the label */
    }
    .fake {
        color: #FF4B4B; /* Streamlit's error color */
        background-color: rgba(255, 75, 75, 0.1);
        border: 1px solid #FF4B4B;
    }
    .true {
        color: #26D367; /* Streamlit's success color */
        background-color: rgba(38, 211, 103, 0.1);
        border: 1px solid #26D367;
    }
</style>
""", unsafe_allow_html=True)



def display_analysis_result(image_file, caption_file):
    with st.container(border=True):
        
        left_col, right_col = st.columns([2, 3])

        with left_col:
            st.subheader("Input Sample")
            st.image(image_file, caption=f"Image: {image_file.name}", use_container_width=True)
            caption_text = caption_file.getvalue().decode("utf-8").strip()
            st.info(caption_text)

        with right_col:
            st.subheader("Analysis Result")
            with st.spinner("Analyzing..."):
                try:
                    image_bytes = image_file.getvalue()
                    files = {"image": (image_file.name, image_bytes, image_file.type)}
                    data = {"caption": caption_text}

                    response = requests.post(BACKEND_URL, files=files, data=data, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        
                        label = result['predicted_label'].upper()
                        style_class = "fake" if label == 'FAKE' else "true"
                        st.markdown(f'<p class="result-label {style_class}">{label}</p>', unsafe_allow_html=True)
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Confidence (for Fake)",
                                value=f"{result['confidence_score']:.4f}"
                            )
                        with metric_col2:
                            st.metric(
                                label="Model Entropy",
                                value=f"{result['entropy_score']:.4f}"
                            )
                        with metric_col3:
                            st.metric(
                                label="Inference Time (s)",
                                value=f"{result['inference_time']:.2f}"
                            )
                    else:
                        st.error(f"API Error (Status {response.status_code}): {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: Could not connect to the backend.")
                    st.info(f"Please ensure the backend server is running at {BACKEND_URL}. Details: {e}")


# --- Main Application UI ---
st.title("Multimodal Fact-Checker 🤖")

st.header("Upload Samples")
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    uploaded_images = st.file_uploader(
        "**1. Choose image files**",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

with upload_col2:
    uploaded_captions = st.file_uploader(
        "**2. Choose caption (.txt) files**",
        type=["txt"],
        accept_multiple_files=True
    )

if st.button("Analyze Samples", type="primary", use_container_width=True):
    if not uploaded_images or not uploaded_captions:
        st.warning("Please upload both image and caption files to proceed.")
    elif len(uploaded_images) != len(uploaded_captions):
        st.error(
            f"File Count Mismatch: You uploaded {len(uploaded_images)} images "
            f"and {len(uploaded_captions)} captions. Please upload an equal number of each."
        )
    else:
        sorted_images = sorted(uploaded_images, key=lambda f: f.name)
        sorted_captions = sorted(uploaded_captions, key=lambda f: f.name)

        st.header("Analysis Results")
        
        for image_file, caption_file in zip(sorted_images, sorted_captions):
            display_analysis_result(image_file, caption_file)
            st.write("") 