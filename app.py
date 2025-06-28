# frontend.py

import streamlit as st
import requests
from PIL import Image
import io


st.set_page_config(
    page_title="RED-DOT Fact-Checker",
    page_icon="🤖",
    layout="wide" 
)

BACKEND_URL = "http://127.0.0.1:8000/predict/"


st.title("RED-DOT Multimodal Fact-Checker 🤖")
st.write(
    "This application uses the RED-DOT model to determine if a claim (caption) about an image is true or fake."
)
st.info(
    "**Instructions:**\n"
    "1. Upload one or more images in the first box.\n"
    "2. Upload the corresponding `.txt` caption files in the second box.\n"
    "3. The number of images must match the number of caption files.\n"
    "4. Click 'Analyze Samples' to see the results."
)


st.header("Upload Your News Samples")


col1, col2 = st.columns(2)

with col1:
   
    uploaded_images = st.file_uploader(
        "1. Choose image files", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

with col2:
    uploaded_captions = st.file_uploader(
        "2. Choose caption (.txt) files", 
        type=["txt"], 
        accept_multiple_files=True
    )

if st.button("Analyze Samples"):
    if not uploaded_images or not uploaded_captions:
        st.warning("Please upload both image and caption files.")
    elif len(uploaded_images) != len(uploaded_captions):
        st.error(
            f"Mismatch in file count. You uploaded {len(uploaded_images)} images "
            f"and {len(uploaded_captions)} captions. Please upload an equal number of each."
        )
    else:
        sorted_images = sorted(uploaded_images, key=lambda f: f.name)
        sorted_captions = sorted(uploaded_captions, key=lambda f: f.name)
        
        st.header("Analysis Results")

     
        for image_file, caption_file in zip(sorted_images, sorted_captions):

            with st.expander(f"Results for: {image_file.name} & {caption_file.name}", expanded=True):
                
               
                caption_text = caption_file.getvalue().decode("utf-8").strip()
                
         
                display_col1, display_col2 = st.columns([1, 2])
                with display_col1:
                    st.image(image_file, caption="Uploaded Image", use_column_width=True)
                with display_col2:
                    st.write("**Input Caption:**")
                    st.info(caption_text)


                with st.spinner("Analyzing..."):
            
                    image_bytes = image_file.getvalue()
                    
                  
                    files = {"image": (image_file.name, image_bytes, image_file.type)}
                    data = {"caption": caption_text}
                    
                    try:
                        response = requests.post(BACKEND_URL, files=files, data=data, timeout=20)

                  
                        if response.status_code == 200:
                            result = response.json()
                            
        
                            st.write("**Model Output:**")
                            
                           
                            if result['predicted_label'].lower() == 'fake':
                                st.error(f"Predicted Label: **{result['predicted_label']}**")
                            else:
                                st.success(f"Predicted Label: **{result['predicted_label']}**")

                           
                            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                            kpi_col1.metric(
                                label="Confidence (for Fake)",
                                value=f"{result['confidence_score']:.4f}"
                            )
                            kpi_col2.metric(
                                label="Model Entropy",
                                value=f"{result['entropy_score']:.4f}"
                            )
                            kpi_col3.metric(
                                label="Inference Time (s)",
                                value=f"{result['inference_time']:.2f}"
                            )
                        else:
                            st.error(f"API Error (Status {response.status_code}): {response.text}")

                    except requests.exceptions.RequestException as e:
                        st.error(f"Failed to connect to the backend API at {BACKEND_URL}.")
                        st.error(f"Error details: {e}")
                        st.info("Please ensure the backend.py server is running.")