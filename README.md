# stratum-ai-dermatology
AI-powered dermatology assistant for skin lesion classification using ConvNeXt and OpenAI CLIP, with FastAPI backend, authentication, admin monitoring, and human-in-the-loop model improvement.

cat > README.md << 'EOF'
# 🧬 Stratum — AI-Powered Skin Analysis

Stratum is a dermatology-assist demo that analyzes skin lesion images using a multi-layer AI approach. It breaks down photos into texture, pigment, and border pattern layers to surface potential conditions worth discussing with a dermatologist.

## ✨ Features

- **Multi-layer analysis**: Surface, texture, pigment, and edge pattern passes
- **7-class classification**: Nevus, Melanoma, BCC, SCC, Actinic Keratosis, Seborrheic Keratosis, Pigmented Benign Keratosis
- **Malignancy risk assessment**: Binary risk prediction
- **User authentication**: Secure login/registration with session management
- **Admin dashboard**: Full analysis visibility, model fine-tuning, and patient management
- **CLIP validator**: Pre-screening to ensure images contain skin lesions
- **Image quality checks**: Brightness, contrast, and detail validation
- **PDF export**: Patient reports and individual analysis exports
- **Feedback system**: Users can correct predictions to improve the model
- **Model versioning**: Fine-tune and promote models with A/B testing

## 🔧 Tech Stack

- **Backend**: FastAPI + PyTorch (ConvNeXt-Tiny)
- **Frontend**: Vanilla JS + Chart.js
- **Database**: SQLite
- **Validation**: OpenAI CLIP
- **Evaluation**: HAM10000 dataset

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stratum.git
cd stratum

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
