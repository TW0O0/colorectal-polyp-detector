
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from streamlit_image_coordinates import streamlit_image_coordinates

# Set page title and configuration
st.set_page_config(page_title="Colorectal Polyp Detector", layout="wide")

# App title and description
st.title("Colorectal Polyp Detector")
st.markdown("""
This application uses a deep learning model to detect and classify colorectal polyps from colonoscopy images.
Upload a clear colonoscopy image for analysis.
""")

# Initialize session state for image handling
if 'image' not in st.session_state:
    st.session_state.image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'crop_coords' not in st.session_state:
    st.session_state.crop_coords = []

# Load the trained model
@st.cache_resource  # Cache the model to avoid reloading
def load_model():
    model_path = os.path.join('model', 'saved_models', 'efficientnet_colorectal_final.h5')
    if not os.path.exists(model_path):
        st.warning(f"Model file not found at {model_path}. Please train the model first.")
        return None
    
    return tf.keras.models.load_model(model_path)

model = load_model()

# Image preprocessing function
def preprocess_image(image, target_size=(224, 224)):
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Image manipulation section
def image_manipulation_section():
    """Function that adds image manipulation capabilities to the Streamlit app."""
    # Only show manipulation tools if an image is uploaded
    if st.session_state.image is None:
        return None
    
    image = st.session_state.image
    
    st.subheader("Image Manipulation")
    st.write("Adjust your colonoscopy image before analysis:")
    
    # Create three columns for different manipulation options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Rotation
        rotation_angle = st.slider("Rotate", -180, 180, 0, 5)
        if rotation_angle != 0:
            image = image.rotate(rotation_angle, expand=True, fillcolor=(0, 0, 0))
    
    with col2:
        # Scaling
        scale_factor = st.slider("Scale", 0.5, 2.0, 1.0, 0.1)
        if scale_factor != 1.0:
            new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
            image = image.resize(new_size, Image.LANCZOS)
    
    with col3:
        # Brightness and contrast
        brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
        contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
        if brightness != 1.0 or contrast != 1.0:
            from PIL import ImageEnhance
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(brightness)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(contrast)
    
    # Cropping section
    st.write("Click and drag on the image to crop (useful for focusing on specific polyp areas):")
    
    # Display image for cropping with coordinate capture
    coords = streamlit_image_coordinates(image, key="crop_tool")
    
    # Handle cropping based on click coordinates
    if coords is not None and coords['x'] is not None:
        if len(st.session_state.crop_coords) < 2:
            st.session_state.crop_coords.append((coords['x'], coords['y']))
            if len(st.session_state.crop_coords) == 1:
                st.write(f"First corner selected at {st.session_state.crop_coords[0]}")
            elif len(st.session_state.crop_coords) == 2:
                st.write(f"Second corner selected at {st.session_state.crop_coords[1]}")
                st.write("Ready to crop! Click 'Apply Crop' to continue.")
    
    # Apply crop button
    if len(st.session_state.crop_coords) == 2 and st.button("Apply Crop"):
        x1, y1 = st.session_state.crop_coords[0]
        x2, y2 = st.session_state.crop_coords[1]
        # Ensure proper coordinate order (top-left, bottom-right)
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
        # Apply the crop
        image = image.crop((left, top, right, bottom))
        # Reset crop coordinates
        st.session_state.crop_coords = []
        st.success("Image cropped successfully!")
    
    # Reset crop button
    if len(st.session_state.crop_coords) > 0 and st.button("Reset Crop Selection"):
        st.session_state.crop_coords = []
        st.info("Crop selection reset")
    
    # Reset all manipulations button
    if st.button("Reset All Manipulations"):
        image = st.session_state.original_image.copy()
        st.session_state.crop_coords = []
        st.info("All manipulations reset")
    
    # Display the manipulated image
    st.image(image, caption="Manipulated Image", use_column_width=True)
    
    # Update the session state with the manipulated image
    st.session_state.image = image
    
    return image

# Image upload
uploaded_file = st.file_uploader("Choose a colonoscopy image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    
    # Store the original image in session state for reset functionality
    if st.session_state.original_image is None or uploaded_file != st.session_state.last_uploaded_file:
        st.session_state.original_image = image.copy()
        st.session_state.image = image.copy()
        st.session_state.last_uploaded_file = uploaded_file
    
    # Display the original uploaded image
    st.image(image, caption='Uploaded Colonoscopy Image', use_column_width=True)
    
    # Add the image manipulation section
    manipulated_image = image_manipulation_section()
    
    # Use the manipulated image for analysis if it exists
    analysis_image = manipulated_image if manipulated_image is not None else image
    
    # Make prediction when the user clicks the button
    if st.button("Analyze Image"):
        if model is not None:
            # Show a spinner while processing
            with st.spinner('Analyzing colonoscopy image...'):
                # Preprocess the image
                processed_img = preprocess_image(analysis_image)
                
                # Make prediction
                predictions = model.predict(processed_img)[0]
                
                # For multi-class classification (normal, benign polyp, potentially cancerous)
                class_names = ["Normal Tissue", "Benign Polyp", "Potentially Cancerous"]
                predicted_class = np.argmax(predictions)
                confidence = predictions[predicted_class] * 100
                
                # Display results
                st.subheader("Analysis Results")
                
                if predicted_class == 0:
                    st.success(f"No polyps detected ({confidence:.1f}% confidence)")
                elif predicted_class == 1:
                    st.warning(f"Benign polyp detected ({confidence:.1f}% confidence)")
                    st.markdown("""
                    **Recommendation**: Routine follow-up recommended. Benign polyps should still 
                    be monitored as they can potentially develop into cancerous lesions over time.
                    """)
                else:
                    st.error(f"Potentially cancerous polyp detected ({confidence:.1f}% confidence)")
                    st.markdown("""
                    **Recommendation**: Immediate consultation with a gastroenterologist is 
                    recommended for further evaluation and potential biopsy.
                    """)
                
                # Show all class probabilities
                st.subheader("Detailed Analysis")
                for i, (class_name, prob) in enumerate(zip(class_names, predictions)):
                    st.text(f"{class_name}: {prob*100:.1f}%")
                    st.progress(float(prob))
                
                # Display disclaimer
                st.warning("Disclaimer: This tool is for research purposes only and should not replace professional medical diagnosis.")
        else:
            st.error("Model not loaded. Please train the model first.")

# Add information about the research project
st.sidebar.title("About")
st.sidebar.info("""
## Colorectal Polyp Detector
This research project aims to develop AI tools for detecting and classifying colorectal polyps from colonoscopy images.
The model uses transfer learning with EfficientNet architecture and was trained on publicly available colonoscopy datasets.

### Datasets Used:
- CVC-ClinicDB
- Kvasir-SEG
- ETIS-LaribPolypDB

For more information or to contribute to this research, please contact research@example.com.
""")

# Add model information
if model is not None:
    st.sidebar.subheader("Model Information")
    st.sidebar.text("Architecture: EfficientNetB3")
    st.sidebar.text("Input size: 224x224 pixels")
    st.sidebar.text("Classes: Normal, Benign, Cancerous")
    st.sidebar.text("Training accuracy: ~91%")

# Add educational information about colorectal polyps
st.sidebar.subheader("Educational Resources")
st.sidebar.markdown("""
### About Colorectal Polyps
Colorectal polyps are growths on the inner lining of the colon or rectum. Most polyps are benign, but some can develop into cancer over time.

#### Key Facts:
- Early detection and removal of polyps can prevent colorectal cancer
- Most polyps don't cause symptoms
- Regular screening is important for adults over 45
- AI-assisted detection can improve polyp detection rates

[Learn more about colorectal cancer screening](https://www.cancer.org/cancer/colon-rectal-cancer/detection-diagnosis-staging/screening-tests-used.html)
""")
