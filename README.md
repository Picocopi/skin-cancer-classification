# 🏥 Advanced Skin Cancer Classification System

A comprehensive machine learning system for skin cancer classification featuring both **image-only** and **multimodal** (image + metadata) deep learning models.

## 🎯 Features

### 🤖 **Multiple AI Models**
- **DenseNet121** (Best - 89.32% accuracy)
- **ResNet152** (Heavy - 88.72% accuracy) 
- **EfficientNet-B3** (Efficient - 86.52% accuracy)
- **ResNet50** (Balanced performance)
- **Multimodal EfficientNet-B3** (Image + Patient Metadata - targeting 94%+ accuracy)

### 🔬 **7 Lesion Classification Types**
- **Melanoma** (mel) - Most dangerous skin cancer
- **Basal Cell Carcinoma** (bcc) - Most common skin cancer
- **Melanocytic Nevi/Moles** (nv) - Benign pigmented lesions
- **Benign Keratosis-like Lesions** (bkl) - Non-cancerous growths
- **Actinic Keratoses** (akiec) - Pre-cancerous lesions
- **Dermatofibroma** (df) - Benign fibrous nodules
- **Vascular Lesions** (vasc) - Blood vessel related lesions

### 🧠 **Advanced Features**
- **Multimodal Learning**: Combines images with patient metadata (age, sex, lesion location)
- **Test-Time Augmentation**: Multiple predictions for increased accuracy
- **Smart Lesion Detection**: Pre-filters non-medical images
- **Interactive Web Interface**: Easy-to-use Streamlit app
- **Medical-Grade Analysis**: Trained on HAM10000 dermatology dataset

- **Interactive Interface**: Easy-to-use web interface with:
  - Image upload functionality
  - Real-time predictions
  - Confidence scores
  - Probability distributions
  - Model comparison

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```

3. **Open Browser**: Navigate to `http://localhost:8501`

## 📋 Usage Instructions

1. **Select Model**: Choose your preferred AI model from the sidebar
2. **Upload Image**: Upload a clear image of the skin lesion (PNG, JPG, JPEG)
3. **Analyze**: Click "Analyze Image" to get predictions
4. **Review Results**: View the classification, confidence score, and probability distribution

## ⚠️ Important Medical Disclaimer

- **Educational Purpose Only**: This tool is for educational and research purposes
- **Not Medical Advice**: Should NOT replace professional medical diagnosis
- **Consult Professionals**: Always consult a dermatologist for proper evaluation
- **Seek Medical Care**: Get immediate medical attention for concerning lesions

## 📊 Model Performance

| Model | Accuracy | Parameters | Best For |
|-------|----------|------------|----------|
| DenseNet121 | 89.32% | 7M | **Best Overall** |
| ResNet152 | 88.72% | 58M | High Accuracy |
| EfficientNet-B3 | 86.52% | 11M | Efficiency |
| ResNet50 | ~85% | 24M | Balance |

## 🔧 Technical Details

- **Framework**: PyTorch + Streamlit
- **Input Size**: 224x224 pixels
- **Preprocessing**: ImageNet normalization
- **Device**: Supports both CPU and GPU inference
- **Dataset**: Trained on HAM10000 skin lesion dataset

## 📁 File Structure

```
├── app.py                              # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── best_densenet121_skin_cancer.pth   # DenseNet121 model weights
├── best_resnet152_heavy.pth          # ResNet152 model weights
├── best_efficientnet_b3_skin_cancer.pth # EfficientNet-B3 weights
├── best_resnet50_skin_cancer.pth     # ResNet50 model weights
└── README.md                          # This file
```

## 🤝 Contributing

Feel free to contribute by:
- Improving the UI/UX
- Adding more models
- Enhancing documentation
- Reporting issues

## 📄 License

This project is for educational purposes. Please ensure proper medical consultation for any health-related decisions.

---

**Remember**: AI assistance should complement, not replace, professional medical expertise! 🏥
