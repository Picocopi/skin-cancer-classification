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

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

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
        
        # 1. Enhanced Skin Color Analysis
        # More flexible skin tone detection with expanded ranges
        skin_ranges = [
            ([0, 20, 40], [25, 255, 255]),      # Light skin (expanded)
            ([0, 15, 30], [30, 200, 255]),      # Medium skin (more flexible)
            ([0, 30, 15], [20, 255, 220]),      # Dark skin (expanded)
            ([5, 40, 60], [35, 180, 255]),      # Additional range for lesions
        ]
        
        total_skin_pixels = 0
        for lower, upper in skin_ranges:
            skin_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            total_skin_pixels += np.sum(skin_mask > 0)
        
        skin_ratio = total_skin_pixels / (224 * 224)
        
        # Detect cartoon/artificial colors (very saturated, unnatural)
        saturation = hsv[:, :, 1]
        high_sat_ratio = np.sum(saturation > 200) / (224 * 224)
        
        # More flexible skin detection for medical images
        if skin_ratio > 0.15:  # Lowered threshold for real lesions
            if high_sat_ratio < 0.4:  # Not too cartoon-like
                checks.append(0.3)
                reasons.append(f"✅ Natural skin/lesion colors detected ({skin_ratio:.1%})")
            else:
                checks.append(0.1)  # Still some credit even if saturated
                reasons.append(f"⚠️ Some saturated colors but possible lesion ({skin_ratio:.1%})")
        elif high_sat_ratio > 0.5:  # Very cartoon-like
            checks.append(-0.3)
            reasons.append(f"❌ Cartoon-like saturated colors detected ({high_sat_ratio:.1%})")
        else:
            checks.append(0.05)  # Small credit for any skin-like colors
            reasons.append(f"⚠️ Limited skin colors but could be dark lesion ({skin_ratio:.1%})")
        
        # 2. Medical Texture Analysis
        # Calculate Local Binary Patterns for texture
        def calculate_lbp(image, radius=1, n_points=8):
            lbp = np.zeros_like(image)
            for i in range(radius, image.shape[0] - radius):
                for j in range(radius, image.shape[1] - radius):
                    center = image[i, j]
                    code = 0
                    for k in range(n_points):
                        angle = 2 * np.pi * k / n_points
                        x = int(i + radius * np.cos(angle))
                        y = int(j + radius * np.sin(angle))
                        if image[x, y] >= center:
                            code |= (1 << k)
                    lbp[i, j] = code
            return lbp
        
        lbp = calculate_lbp(gray_resized)
        lbp_variance = np.var(lbp)
        
        # Medical skin images have specific texture variance ranges
        if 100 < lbp_variance < 3000:  # More flexible range for real lesions
            checks.append(0.2)
            reasons.append(f"✅ Medical texture patterns detected (LBP: {lbp_variance:.0f})")
        elif lbp_variance < 50:  # Very low = likely artificial
            checks.append(-0.1)
            reasons.append(f"❌ Too uniform texture (LBP: {lbp_variance:.0f})")
        else:
            checks.append(0.05)  # Small credit for any texture
            reasons.append(f"⚠️ Unusual but possible texture pattern (LBP: {lbp_variance:.0f})")
        
        # 3. Edge and Gradient Analysis
        sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Check for sharp artificial edges (cartoons have very sharp edges)
        sharp_edges = np.sum(gradient_magnitude > 100)
        edge_ratio = sharp_edges / (224 * 224)
        
        if edge_ratio > 0.2:  # Very sharp edges = likely cartoon/artificial
            checks.append(-0.25)
            reasons.append(f"❌ Too many sharp artificial edges ({edge_ratio:.1%})")
        elif 0.01 < edge_ratio < 0.15:  # More flexible range for medical
            checks.append(0.15)
            reasons.append(f"✅ Natural edge distribution ({edge_ratio:.1%})")
        else:
            checks.append(0.05)  # Small credit for any edges
            reasons.append(f"⚠️ Edge pattern acceptable ({edge_ratio:.1%})")
        
        # 4. Color Distribution Analysis
        # Medical images have more muted, realistic color distributions
        b, g, r = cv2.split(image_resized)
        
        # Check for unrealistic color combinations
        color_std = (np.std(r) + np.std(g) + np.std(b)) / 3
        color_mean_diff = abs(np.mean(r) - np.mean(g)) + abs(np.mean(g) - np.mean(b))
        
        if color_std > 70:  # Very high color variation = likely cartoon
            checks.append(-0.2)
            reasons.append(f"❌ Excessive color variation (std: {color_std:.0f})")
        elif 15 < color_std < 60:  # More flexible range for medical
            checks.append(0.15)
            reasons.append(f"✅ Natural color variation (std: {color_std:.0f})")
        else:
            checks.append(0.05)  # Small credit for any variation
            reasons.append(f"⚠️ Color variation acceptable (std: {color_std:.0f})")
        
        # 5. Shape and Object Detection
        # Use contour detection to identify shapes
        edges = cv2.Canny(gray_resized, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Check for geometric shapes (common in cartoons)
        geometric_shapes = 0
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) < 8:  # Geometric shapes have few vertices
                    geometric_shapes += 1
        
        if geometric_shapes > 5:  # Many geometric shapes = likely cartoon
            checks.append(-0.3)
            reasons.append(f"❌ Multiple geometric shapes detected ({geometric_shapes})")
        elif geometric_shapes <= 2:  # Few shapes = good for medical
            checks.append(0.1)
            reasons.append("✅ No obvious geometric shapes")
        else:
            checks.append(0.0)
            reasons.append(f"⚠️ Some geometric elements but acceptable ({geometric_shapes})")
        
        # 6. Face Detection (reject if face is detected)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_resized, 1.1, 4)
        
        if len(faces) > 0:
            checks.append(-0.4)
            reasons.append(f"❌ Face detected - not a skin lesion ({len(faces)} faces)")
        else:
            checks.append(0.1)
            reasons.append("✅ No faces detected")
        
        # 7. Text Detection (reject if text is detected)
        # Simple text detection using edge density in horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        detected_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, horizontal_kernel)
        text_pixels = np.sum(detected_lines > 0)
        text_ratio = text_pixels / (224 * 224)
        
        if text_ratio > 0.15:  # Significant text = reject
            checks.append(-0.3)
            reasons.append(f"❌ Text-like patterns detected ({text_ratio:.1%})")
        else:
            checks.append(0.05)
            reasons.append("✅ No significant text patterns detected")
        
        # 8. Medical Image Characteristics
        # Check for circular/oval lesion-like shapes
        circles = cv2.HoughCircles(gray_resized, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=10, maxRadius=100)
        
        if circles is not None and len(circles[0]) > 0:
            checks.append(0.2)
            reasons.append(f"✅ Circular lesion-like shapes detected ({len(circles[0])})")
        else:
            checks.append(0.0)
            reasons.append("⚠️ No obvious circular lesions")
        
        # --- REMOVED RASH DETECTION LOGIC ---
        # The following checks for rashes, multiple lesions, and inflammatory
        # patterns have been removed to allow analysis of such images.
        
        # Calculate final confidence with more balanced thresholds
        total_confidence = max(0.0, min(1.0, sum(checks)))
        is_lesion = total_confidence > 0.5  # Slightly higher threshold due to additional checks
        
        return is_lesion, total_confidence, reasons
        
    except Exception as e:
        return False, 0.0, [f"❌ Error analyzing image: {str(e)}"]

@st.cache_resource
def load_model(model_name, model_path):
    """Load the trained model"""
    try:
        # --- Model Architectures ---
        if model_name == "DenseNet121":
            model = models.densenet121(weights=None)
            # This model was likely trained with a simple linear layer replacement
            model.classifier = nn.Linear(model.classifier.in_features, len(CLASS_NAMES))

        elif model_name == "ResNet152":
            model = models.resnet152(weights=None)
            # ResNet152 was trained with dropout layer - match the training architecture
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, len(CLASS_NAMES))
            )

        elif model_name == "EfficientNet-B3":
            model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=len(CLASS_NAMES))
            # Replicate the architecture from the training script
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                dropout_rate = 0.4 # Use the same dropout as in training
                model.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate, inplace=True),
                    model.classifier # The original nn.Linear layer
                )

        elif model_name == "ResNet50":
            model = models.resnet50(weights=None)
            # ResNet50 might have been trained with or without dropout - try both
            try:
                # First try simple linear layer
                model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
                # Test load to see if this works
                if os.path.exists(model_path):
                    test_state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(test_state_dict)
            except:
                # If that fails, try with dropout layer
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
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
    """
    Predict skin lesion class from an image using Test-Time Augmentation (TTA).
    """
    model.eval()

    # Define TTA transforms
    tta_transforms = [
        transforms.Compose([ # Original
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([ # Horizontal Flip
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        transforms.Compose([ # Center Crop from larger resize
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

# Available models
AVAILABLE_MODELS = {
    "DenseNet121 (Best - 89.32%)": "best_densenet121_skin_cancer.pth",
    "EfficientNet-B3 (Real-World Trained)": "best_efficientnet_b3_realworld.pth",
    "ResNet152 (Heavy - 88.72%)": "best_resnet152_heavy.pth", 
    "EfficientNet-B3 (Efficient - 86.52%)": "best_efficientnet_b3_skin_cancer.pth",
    "ResNet50 (Balanced)": "best_resnet50_skin_cancer.pth"
}

# Multimodal model class
class MultiModalModel(nn.Module):
    """
    Multimodal model combining EfficientNet-B3 for images and metadata features
    """
    def __init__(self, num_classes=7, metadata_dim=8, dropout_rate=0.4):
        super(MultiModalModel, self).__init__()
        
        # Image encoder - EfficientNet-B3
        self.image_encoder = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
        image_features = self.image_encoder.num_features  # 1536 for EfficientNet-B3
        
        # Metadata encoder
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Combined classifier
        combined_features = image_features + 32  # 1536 + 32 = 1568
        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, metadata):
        # Extract image features
        image_features = self.image_encoder(image)
        
        # Extract metadata features
        metadata_features = self.metadata_encoder(metadata)
        
        # Combine features
        combined_features = torch.cat([image_features, metadata_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        return output

def encode_metadata(age, sex, lesion_location):
    """
    Encode metadata features the same way as in training
    """
    # Age normalization (0-100 years)
    age_normalized = age / 100.0
    
    # Sex encoding
    sex_male = 1.0 if sex.lower() == 'male' else 0.0
    sex_female = 1.0 if sex.lower() == 'female' else 0.0
    
    # Location encoding (one-hot)
    location_mapping = {
        'scalp': 0, 'ear': 1, 'face': 2, 'back': 3, 'trunk': 4, 
        'chest': 5, 'upper extremity': 6, 'abdomen': 7, 'lower extremity': 8,
        'genital': 9, 'hand': 10, 'foot': 11, 'nail': 12, 'neck': 13, 'unknown': 14
    }
    
    location_encoded = [0.0] * 5  # Use top 5 most common locations
    common_locations = ['back', 'lower extremity', 'trunk', 'upper extremity', 'face']
    
    if lesion_location.lower() in common_locations:
        idx = common_locations.index(lesion_location.lower())
        location_encoded[idx] = 1.0
    
    # Combine all metadata features
    metadata_features = [age_normalized, sex_male, sex_female] + location_encoded
    return torch.tensor(metadata_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

# Main app
def main():
    st.title("🏥 Skin Cancer Classification App")
    st.markdown("Upload a skin lesion image to get an AI-powered classification")
    
    # Sidebar for model selection
    st.sidebar.header("🤖 Model Selection")
    selected_model_display = st.sidebar.selectbox(
        "Choose a model:",
        list(AVAILABLE_MODELS.keys()),
        index=0  # Default to DenseNet121
    )
    
    # Get actual model name and path
    model_path = AVAILABLE_MODELS[selected_model_display]
    model_architecture = selected_model_display.split(" ")[0] # e.g., "EfficientNet-B3"
    
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
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Metadata inputs for multimodal model
    if is_multimodal:
        with col2:
            st.header("� Patient Metadata")
            st.info("📝 Please provide patient information for enhanced analysis")
            
            # Age input
            age = st.number_input(
                "Age (years)",
                min_value=0,
                max_value=100,
                value=45,
                help="Patient's age in years"
            )
            
            # Sex input
            sex = st.selectbox(
                "Sex",
                options=["Male", "Female"],
                help="Patient's biological sex"
            )
            
            # Lesion location input
            lesion_location = st.selectbox(
                "Lesion Location",
                options=[
                    "Back",
                    "Lower extremity", 
                    "Trunk",
                    "Upper extremity",
                    "Face",
                    "Scalp",
                    "Ear",
                    "Chest",
                    "Abdomen",
                    "Hand",
                    "Foot",
                    "Neck",
                    "Genital",
                    "Nail",
                    "Unknown"
                ],
                help="Anatomical location of the lesion"
            )
            
            st.markdown("---")
            st.markdown("**� Current Metadata:**")
            st.write(f"• **Age:** {age} years")
            st.write(f"• **Sex:** {sex}")
            st.write(f"• **Location:** {lesion_location}")
    
    # Results column
    results_col = col3 if is_multimodal else col2
    
    with results_col:
        st.header("📋 Results")
        
        if uploaded_file is not None:
            # First, check if this is likely a skin lesion
            with st.spinner("🔍 Checking if image contains a skin lesion..."):
                is_lesion, confidence, reasons = is_skin_lesion(image)
                
                # Display detection results
                st.subheader("🎯 Skin Lesion Detection")
                
                if is_lesion:
                    st.success(f"✅ **Likely contains a skin lesion** (Score: {confidence:.2f}/1.0)")
                    
                    # Show detection details in an expander
                    with st.expander("🔍 Detection Details"):
                        for reason in reasons:
                            st.write(f"• {reason}")
                    
                    # Proceed with classification
                    analyze_button_text = "🔍 Analyze with Image & Metadata" if is_multimodal else "🔍 Analyze Skin Lesion"
                    
                    if st.button(analyze_button_text, type="primary"):
                        with st.spinner("Analyzing skin lesion..."):
                            if is_multimodal:
                                predicted_class, probabilities = predict_multimodal(
                                    model, image, age, sex, lesion_location
                                )
                            else:
                                predicted_class, probabilities = predict_image(model, image)
                        
                        if predicted_class is not None:
                            # Store results in session state
                            st.session_state.prediction_results = {
                                'predicted_class': predicted_class,
                                'probabilities': probabilities,
                                'model_name': selected_model_display,
                                'is_valid_lesion': True,
                                'is_multimodal': is_multimodal,
                                'metadata': {
                                    'age': age,
                                    'sex': sex,
                                    'location': lesion_location
                                } if is_multimodal else None
                            }
                else:
                    st.error(f"❌ **Unlikely to be a skin lesion** (Score: {confidence:.2f}/1.0)")
                    
                    st.warning("**⚠️ Please upload a clear image of a single skin lesion for accurate analysis.**")
                    
                    # Show what went wrong
                    with st.expander("❌ Issues Detected"):
                        for reason in reasons:
                            st.write(f"• {reason}")
                    
                    # Helpful tips
                    st.info("""
                    **💡 Tips for better results:**
                    - Ensure the image shows a skin lesion, mole, or suspicious spot
                    - Use good lighting and avoid shadows
                    - Make sure the lesion is the main focus of the image  
                    - Avoid images of objects, text, faces, or non-medical content
                    - Take photos at close range showing skin texture clearly
                    """)
                    
                    # Clear any previous results
                    if hasattr(st.session_state, 'prediction_results'):
                        st.session_state.prediction_results = {
                            'is_valid_lesion': False
                        }
    
    # Display prediction results (in a new row if multimodal)
    if is_multimodal:
        st.markdown("---")
        st.header("🎯 Analysis Results")
    
    if hasattr(st.session_state, 'prediction_results') and st.session_state.prediction_results.get('is_valid_lesion', False):
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
            
            # Create a nice probability display
            prob_data = []
            for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
                prob_percentage = prob * 100
                prob_data.append({
                    'Class': class_name.upper(),
                    'Description': CLASS_DESCRIPTIONS[class_name],
                    'Probability': f"{prob_percentage:.1f}%"
                })
            
            # Display as a table
            import pandas as pd
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
            
        elif hasattr(st.session_state, 'prediction_results') and not st.session_state.prediction_results.get('is_valid_lesion', True):
            st.info("❌ **No valid skin lesion detected in uploaded image**")
            st.markdown("""
            Please upload a clear image of a skin lesion for analysis.
            
            **What we're looking for:**
            - Close-up photos of moles, spots, or lesions on skin
            - Clear, well-lit medical images
            - Images where the skin lesion is the main subject
            """)
        else:
            st.info("👆 Upload an image above to see the analysis results")
    
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
