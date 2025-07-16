import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Skin Cancer Classification",
    page_icon="🏥",
    layout="wide"
)

# Define class names
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_DESCRIPTIONS = {
    'akiec': 'Actinic Keratoses (Solar Keratoses)',
    'bcc': 'Basal Cell Carcinoma', 
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi (Moles)',
    'vasc': 'Pyogenic Granulomas and Hemorrhage'
}

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Available models (removed multimodal for now)
AVAILABLE_MODELS = {
    "DenseNet121 (Best - 89.32%)": "best_densenet121_skin_cancer.pth",
    "EfficientNet-B3 (Real-World Trained)": "best_efficientnet_b3_realworld.pth",
    "ResNet152 (Heavy - 88.72%)": "best_resnet152_heavy.pth", 
    "EfficientNet-B3 (Efficient - 86.52%)": "best_efficientnet_b3_skin_cancer.pth",
    "ResNet50 (Balanced)": "best_resnet50_skin_cancer.pth"
}

def is_skin_lesion(image_pil):
    """
    Advanced detection of skin lesions using multiple computer vision techniques
    Returns: (is_lesion, confidence, reasons)
    """
    try:
        # Convert PIL to OpenCV format
        image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        image_gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent analysis
        image_resized = cv2.resize(image_cv, (224, 224))
        gray_resized = cv2.resize(image_gray, (224, 224))
        hsv = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        
        checks = []
        reasons = []
        
        # 1. Enhanced Skin Color Analysis (including lesion colors)
        skin_ranges = [
            ([0, 20, 40], [25, 255, 255]),      # Light skin tones
            ([0, 15, 30], [30, 200, 255]),      # Medium skin tones
            ([0, 30, 15], [20, 255, 220]),      # Dark skin tones
            ([5, 40, 60], [35, 180, 255]),      # General lesion colors
        ]
        
        # Additional ranges for vascular and pigmented lesions
        lesion_ranges = [
            ([0, 80, 80], [15, 255, 255]),      # Red/pink vascular lesions
            ([110, 50, 50], [130, 255, 255]),   # Blue/purple vascular lesions
            ([15, 30, 30], [35, 255, 200]),     # Brown pigmented lesions
            ([0, 0, 20], [180, 50, 100]),       # Dark/black lesions
        ]
        
        total_skin_pixels = 0
        total_lesion_pixels = 0
        
        # Check for skin-like colors
        for lower, upper in skin_ranges:
            skin_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_skin_pixels += np.sum(skin_mask > 0)
        
        # Check for lesion-specific colors
        for lower, upper in lesion_ranges:
            lesion_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_lesion_pixels += np.sum(lesion_mask > 0)
        
        skin_ratio = total_skin_pixels / (224 * 224)
        lesion_ratio = total_lesion_pixels / (224 * 224)
        combined_ratio = skin_ratio + lesion_ratio
        
        # More inclusive scoring for medical images
        if combined_ratio > 0.20:  # Lowered threshold
            checks.append(0.4)
            reasons.append(f"✅ Medical skin/lesion colors detected ({combined_ratio:.1%})")
        elif combined_ratio > 0.10:
            checks.append(0.3)
            reasons.append(f"✅ Possible lesion colors detected ({combined_ratio:.1%})")
        elif lesion_ratio > 0.05:  # Special case for dark/vascular lesions
            checks.append(0.25)
            reasons.append(f"✅ Vascular/pigmented lesion colors detected ({lesion_ratio:.1%})")
        else:
            checks.append(0.1)
            reasons.append(f"⚠️ Limited typical colors but could be unusual lesion ({combined_ratio:.1%})")
        
        # 2. Texture analysis - look for medical image characteristics
        # Calculate image variance (medical images have specific texture ranges)
        gray_var = np.var(gray_resized)
        if 100 < gray_var < 5000:  # Medical images typically in this range
            checks.append(0.2)
            reasons.append(f"✅ Medical image texture detected (variance: {gray_var:.0f})")
        else:
            checks.append(0.1)
            reasons.append(f"⚠️ Unusual texture but could be lesion (variance: {gray_var:.0f})")
        
        # 3. Check for obvious non-medical content
        # Look for very uniform colors (like screenshots, logos)
        unique_colors = len(np.unique(image_resized.reshape(-1, 3), axis=0))
        if unique_colors < 100:  # Very few colors = likely not medical
            checks.append(-0.2)
            reasons.append(f"❌ Too few colors for medical image ({unique_colors} unique colors)")
        elif unique_colors > 1000:  # Rich color variation = likely medical
            checks.append(0.15)
            reasons.append(f"✅ Rich color variation typical of medical images ({unique_colors} colors)")
        else:
            checks.append(0.05)
            reasons.append(f"⚠️ Moderate color variation ({unique_colors} colors)")
        
        # Calculate final confidence with lower threshold for medical images
        total_confidence = max(0.0, min(1.0, sum(checks)))
        is_lesion = total_confidence > 0.3  # Lowered from 0.5 to be more inclusive
        
        return is_lesion, total_confidence, reasons
        
    except Exception as e:
        return False, 0.0, [f"❌ Error analyzing image: {str(e)}"]

@st.cache_resource
def load_model(model_name, model_path):
    """Load the trained model"""
    try:
        if model_name == "DenseNet121":
            model = models.densenet121(weights=None)
            # The saved model has a Sequential classifier with dropout
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
            )
        elif model_name == "ResNet152":
            model = models.resnet152(weights=None)
            # The saved model has a Sequential fc layer with dropout
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            )
        elif model_name == "EfficientNet-B3":
            model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(CLASS_NAMES))
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                dropout_rate = 0.4
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate, inplace=True),
                    model.classifier
                )
        elif model_name == "ResNet50":
            model = models.resnet50(weights=None)
            # The saved model has a Sequential fc layer with dropout
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            )
        else:
            st.error(f"Unknown model architecture: {model_name}")
            return None
            
        # Load state dict
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_image(model, image):
    """Predict skin lesion class from an image using Test-Time Augmentation"""
    model.eval()

    # Define TTA transforms
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]

    all_probs = []
    with torch.no_grad():
        for tta_transform in tta_transforms:
            augmented_image = tta_transform(image).unsqueeze(0).to(device)
            outputs = model(augmented_image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probabilities.cpu().numpy())

    # Average the probabilities from all augmented images
    avg_probs = np.mean(all_probs, axis=0).flatten()
    predicted_class_idx = np.argmax(avg_probs)
    return predicted_class_idx, avg_probs

def main():
    st.title("🏥 Skin Cancer Classification App")
    st.markdown("Upload a skin lesion image to get an AI-powered classification")
    
    # Sidebar for model selection
    st.sidebar.header("🤖 Model Selection")
    selected_model_display = st.sidebar.selectbox(
        "Choose a model:",
        list(AVAILABLE_MODELS.keys()),
        index=0
    )
    
    # Get actual model name and path
    model_path = AVAILABLE_MODELS[selected_model_display]
    model_architecture = selected_model_display.split(" ")[0]
    
    st.sidebar.markdown("### 📊 Model Performance:")
    performance_info = {
        "EfficientNet-B3 (Real-World Trained)": "Trained on real-world augmentations for better generalization.",
        "DenseNet121 (Best - 89.32%)": "89.32% accuracy, 7M params",
        "ResNet152 (Heavy - 88.72%)": "88.72% accuracy, 58M params", 
        "EfficientNet-B3 (Efficient - 86.52%)": "86.52% accuracy, 11M params",
        "ResNet50 (Balanced)": "~85% accuracy, 24M params"
    }
    st.sidebar.info(performance_info.get(selected_model_display, "Performance info not available"))
    
    # Load selected model
    with st.spinner(f"Loading {selected_model_display} model..."):
        model = load_model(model_architecture, model_path)
    
    if model is not None:
        st.success(f"✅ {selected_model_display} model loaded successfully!")
    else:
        st.error("❌ Failed to load model. Please check if model files exist.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin lesion image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin lesion for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Results column
    with col2:
        st.header("📋 Results")
        
        if uploaded_file is not None:
            # First, check if this is likely a skin lesion
            with st.spinner("🔍 Checking if image contains a skin lesion..."):
                is_lesion, confidence, reasons = is_skin_lesion(image)
                
                st.subheader("🎯 Skin Lesion Detection")
                
                if is_lesion:
                    st.success(f"✅ **Likely contains a skin lesion** (Score: {confidence:.2f}/1.0)")
                    
                    with st.expander("🔍 Detection Details"):
                        for reason in reasons:
                            st.write(f"• {reason}")
                    
                    if st.button("🔍 Analyze Skin Lesion", type="primary"):
                        with st.spinner("Analyzing skin lesion..."):
                            predicted_class, probabilities = predict_image(model, image)
                        
                        if predicted_class is not None:
                            st.session_state.prediction_results = {
                                'predicted_class': predicted_class,
                                'probabilities': probabilities,
                                'model_name': selected_model_display,
                                'is_valid_lesion': True
                            }
                else:
                    st.error(f"❌ **Unlikely to be a skin lesion** (Score: {confidence:.2f}/1.0)")
                    st.warning("**⚠️ Please upload a clear image of a single skin lesion for accurate analysis.**")
                    
                    with st.expander("❌ Issues Detected"):
                        for reason in reasons:
                            st.write(f"• {reason}")
    
    # Display prediction results
    if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results.get('is_valid_lesion', False):
        st.markdown("---")
        st.header("🎯 Analysis Results")
        
        results = st.session_state.prediction_results
        predicted_class = results['predicted_class']
        probabilities = results['probabilities']
        model_name = results['model_name']
        
        # Display prediction
        predicted_label = CLASS_NAMES[predicted_class]
        confidence = probabilities[predicted_class] * 100
        
        # Color code based on confidence
        if confidence > 80:
            confidence_color = "🟢"
        elif confidence > 60:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        st.markdown(f"### 🎯 Prediction: **{predicted_label.upper()}**")
        st.markdown(f"### {confidence_color} Confidence: **{confidence:.1f}%**")
        st.markdown(f"**Description:** {CLASS_DESCRIPTIONS[predicted_label]}")
        
        # Show all probabilities
        st.markdown("### 📊 All Class Probabilities:")
        
        prob_data = []
        for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
            prob_percentage = prob * 100
            prob_data.append({
                'Class': class_name.upper(),
                'Description': CLASS_DESCRIPTIONS[class_name],
                'Probability': f"{prob_percentage:.1f}%"
            })
        
        df = pd.DataFrame(prob_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization of probabilities
        st.markdown("### 📈 Probability Distribution:")
        chart_data = pd.DataFrame({
            'Classes': [name.upper() for name in CLASS_NAMES],
            'Probabilities': probabilities * 100
        })
        st.bar_chart(chart_data.set_index('Classes'))
        
        # Important disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Important Medical Disclaimer:**
        - This AI tool is for educational purposes only
        - It should NOT replace professional medical diagnosis
        - Always consult a dermatologist for proper medical evaluation
        - Seek immediate medical attention for concerning lesions
        """)
    
    # Information section
    st.markdown("---")
    st.header("ℹ️ About This App")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.markdown("""
        **🎯 What it does:**
        - Classifies skin lesions into 7 categories
        - Uses state-of-the-art deep learning models
        - Provides confidence scores for predictions
        """)
    
    with info_col2:
        st.markdown("""
        **🔬 Lesion Types:**
        - Melanoma (mel)
        - Basal Cell Carcinoma (bcc) 
        - Benign lesions (nv, bkl, df)
        - Solar keratoses (akiec)
        - Vascular lesions (vasc)
        """)
    
    with info_col3:
        st.markdown("""
        **🤖 Available Models:**
        - DenseNet121: Best accuracy (89.32%)
        - ResNet152: Heavy but accurate (88.72%)
        - EfficientNet-B3: Efficient (86.52%)
        - ResNet50: Balanced performance
        """)

if __name__ == "__main__":
    main()
