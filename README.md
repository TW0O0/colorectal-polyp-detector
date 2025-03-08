# Colorectal Polyp Detector

This project is a research application that uses deep learning to detect and classify colorectal polyps from colonoscopy images. It demonstrates how a simple AI-powered medical diagnostic tool can be built using TensorFlow and deployed with Streamlit.

## Features

- Upload and analyze colonoscopy images
- Image manipulation tools (crop, rotate, adjust brightness/contrast)
- Multi-class classification of colorectal tissues:
  - Normal colon tissue
  - Benign polyps
  - Potentially cancerous polyps
- Detailed probability analysis
- Educational resources about colorectal cancer

## Project Structure

```
colorectal-polyp-detector/
├── app.py                  # Streamlit web application
├── train_model.py          # Script for training the model
├── requirements.txt        # Python dependencies
├── model/                  # Directory for model storage
│   └── saved_models/       # Saved model files
├── data/                   # Dataset directory (not included in repo)
│   └── colorectal_dataset/ # Organized dataset
│       ├── train/          # Training images
│       ├── val/            # Validation images
│       └── test/           # Test images
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/username/colorectal-polyp-detector.git
   cd colorectal-polyp-detector
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download and organize datasets (not included in this repository due to size and licensing):
   - Download datasets from their sources:
     - [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/)
     - [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
     - [ETIS-LaribPolypDB](https://polyp.grand-challenge.org/EtisLarib/)
   - Organize them into train/val/test folders with class subdirectories:
     ```
     data/colorectal_dataset/
     ├── train/
     │   ├── normal/
     │   ├── benign/
     │   └── cancerous/
     ├── val/
     │   ├── normal/
     │   ├── benign/
     │   └── cancerous/
     └── test/
         ├── normal/
         ├── benign/
         └── cancerous/
     ```

### Training the Model

1. Run the training script:
   ```
   python train_model.py
   ```

2. The trained model will be saved to `model/saved_models/efficientnet_colorectal_final.h5`

### Running the Application

1. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

## Using the Application

1. Upload a colonoscopy image using the file uploader
2. Use the image manipulation tools if needed
3. Click "Analyze Image" to get the prediction results
4. Review the detailed analysis and recommendations

## Model Details

- Architecture: EfficientNetB3 with transfer learning
- Input size: 224x224 pixels RGB images
- Output: 3 classes (Normal, Benign Polyp, Potentially Cancerous Polyp)
- Training approach: Two-phase training with frozen base model followed by fine-tuning

## Research Application

This tool is designed for research purposes only and should not be used for clinical diagnosis. If you're interested in contributing to or extending this research, consider:

1. Improving the model with additional datasets
2. Adding segmentation capabilities to highlight polyp regions
3. Implementing explainable AI techniques to visualize model decision-making
4. Conducting clinical validation studies with medical professionals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses public datasets created by medical research institutions
- Built with TensorFlow and Streamlit open-source libraries
