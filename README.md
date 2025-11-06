# AI Review Detection: ICLR Paper Review Analysis

This project implements a complete pipeline for detecting AI-generated paper reviews using fine-tuned language models. The system downloads ICLR conference data, generates synthetic reviews using AI models, trains a classifier to distinguish between real and AI-generated reviews, and performs inference on new data.

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Workflow](#pipeline-workflow)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Download ICLR Data](#1-download-iclr-data)
  - [2. Generate AI Reviews](#2-generate-ai-reviews)
  - [3. Fine-tune Classification Model](#3-fine-tune-classification-model)
  - [4. Inference on New Data](#4-inference-on-new-data)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project addresses the growing concern of AI-generated academic reviews by:

1. **Collecting real reviews** from ICLR (International Conference on Learning Representations)
2. **Generating synthetic reviews** using state-of-the-art LLMs (GPT-4o, DeepSeek Reasoner)
3. **Training a classifier** using Longformer with LoRA fine-tuning to detect AI-generated content
4. **Analyzing trends** in AI-generated reviews across multiple years (2021-2023)

## 🔄 Pipeline Workflow

```
┌─────────────────────┐
│  1. Download Data   │  ← download_review.ipynb
│  (ICLR Papers +     │
│   Real Reviews)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. Generate AI     │  ← generate_review.ipynb
│     Reviews         │
│  (GPT-4o/DeepSeek)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. Fine-tune       │  ← finetune_lm.ipynb
│     Classifier      │
│  (Longformer+LoRA)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. Inference       │  ← inference.ipynb
│  (Classify Reviews) │
└─────────────────────┘
```

## 📦 Requirements

- Python 3.8+
- Google Colab (recommended) or local GPU environment
- NVIDIA GPU with CUDA support (for training and inference)
- API keys for OpenAI or DeepSeek (for review generation)

### Python Dependencies

```bash
openreview-py
requests
openai
PyPDF2
transformers
datasets
accelerate
peft
torch
pandas
tqdm
scikit-learn
```

## 🚀 Installation

All notebooks are designed to run on Google Colab with automatic dependency installation. Each notebook includes installation cells at the beginning.

For local setup:

```bash
pip install openreview-py requests openai PyPDF2
pip install transformers datasets accelerate peft
pip install torch pandas tqdm scikit-learn
```

## 📖 Usage

### 1. Download ICLR Data

**Notebook:** `download_review.ipynb`

This notebook downloads papers and official reviews from ICLR conferences.

**Features:**
- Downloads paper PDFs for specified years
- Retrieves official reviews from OpenReview API
- Supports both API v1 (2020-2023) and v2 (2024+)
- Saves data in organized CSV format

**Configuration:**

```python
YEARS = [2020, 2021, 2022, 2023]  # Years to download
NUM_PAPERS = 100  # Number of papers per year (None for all)
```

**Output:**
- `iclr_YEAR_data/pdfs/` - Downloaded paper PDFs
- `iclr_YEAR_data/iclr_YEAR_reviews.csv` - Review data with paper titles

**Steps:**
1. Mount Google Drive
2. Configure years and number of papers
3. Run download function
4. Data will be saved to `Google Drive/Notebooks/AI_review/`

---

### 2. Generate AI Reviews

**Notebook:** `generate_review.ipynb`

Generates synthetic reviews using GPT-4o or DeepSeek Reasoner API.

**Features:**
- Extracts text from PDF papers
- Generates ICLR-style reviews using AI models
- Batch processes multiple papers
- Saves results in CSV format matching real review structure

**Supported APIs:**
- OpenAI GPT-4o (`api_name="gpt"`)
- DeepSeek Reasoner (`api_name="deepseek"`)

**Configuration:**

```python
API_KEY = "your-api-key-here"
YEAR = 2020  # Must match downloaded data year
API_NAME = "deepseek"  # or "gpt"
PDF_FOLDER = f"{WORK_DIR}/iclr_{YEAR}_data/pdfs"
OUTPUT_CSV = f"{WORK_DIR}/ai_review_{YEAR}.csv"
```

**Prompt Design:**
The system uses a carefully crafted prompt to generate natural, professional reviews that include:
- Paper summary and contributions
- Strengths and weaknesses discussion
- Technical questions and concerns
- Comments on clarity, novelty, and experiments
- Overall assessment (200-400 words)

**Output:**
- `ai_review_YEAR.csv` - Generated reviews with paper titles

---

### 3. Fine-tune Classification Model

**Notebook:** `finetune_lm.ipynb`

Trains a binary classifier to distinguish between real and AI-generated reviews using Longformer with LoRA.

**Model Architecture:**
- Base: `allenai/longformer-base-4096` (supports long documents up to 4096 tokens)
- Fine-tuning: LoRA (Low-Rank Adaptation) for memory efficiency
- Task: Binary sequence classification

**Why Longformer + LoRA?**
- **Longformer**: Handles long review texts (avg. 561 tokens, max 1649 tokens)
- **LoRA**: Reduces trainable parameters from 149M to 887K (0.59%)
- **Memory efficient**: Enables training on consumer GPUs

**Configuration:**

```python
REAL_REVIEW_PATH = 'path/to/iclr_2020_reviews.csv'  # Label 0
AI_REVIEW_PATH = 'path/to/ai_review_2020.csv'       # Label 1
OUTPUT_DIR = 'path/to/save/model'

MAX_LENGTH = 2048
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-4

# LoRA parameters
LORA_R = 8          # LoRA rank
LORA_ALPHA = 16     # 2 * rank
LORA_DROPOUT = 0.1
```

**Training Features:**
- Automatic data balancing (equal samples per class)
- Mixed precision training (FP16)
- Gradient checkpointing for memory efficiency
- Gradient accumulation
- Stratified train/validation split (80/20)

**Output:**
- Fine-tuned model saved to Google Drive
- LoRA adapter weights (very small, ~3.5MB)
- Tokenizer configuration

**Expected Performance:**
- Training completes in ~10-15 minutes on L4 GPU
- Evaluation loss: ~0.64
- Model distinguishes patterns in review writing styles

---

### 4. Inference on New Data

**Notebook:** `inference.ipynb`

Applies the trained classifier to predict whether reviews are real or AI-generated.

**Features:**
- Loads fine-tuned LoRA model
- Batch inference on multiple years
- Generates detailed statistics and confidence scores
- Saves results with predictions and probabilities

**Configuration:**

```python
MODEL_PATH = 'path/to/finetuned_longformer_lora'
BASE_MODEL_NAME = 'allenai/longformer-base-4096'

DATA_PATHS = {
    '2021': 'path/to/iclr_2021_reviews.csv',
    '2022': 'path/to/iclr_2022_reviews.csv',
    '2023': 'path/to/iclr_2023_reviews.csv',
}

MAX_LENGTH = 2048
BATCH_SIZE = 8
```

**Output Files:**

1. **Individual Year Results** (`inference_results_YEAR_timestamp.csv`):
   - Original paper title and review text
   - `predicted_label`: 0 (Real) or 1 (AI-generated)
   - `predicted_class`: "Real" or "AI-generated"
   - `prob_real`: Confidence score for real review
   - `prob_ai`: Confidence score for AI-generated

2. **Summary Statistics** (`summary_YEAR_timestamp.csv`):
   - Total reviews analyzed
   - Count and percentage of real vs AI-generated

3. **Overall Summary** (`overall_summary_timestamp.csv`):
   - Cross-year comparison
   - AI-generation trends over time

**Example Results:**

| Year | Total Reviews | AI Count | AI Percentage |
|------|--------------|----------|---------------|
| 2021 | 388          | 264      | 68.04%        |
| 2022 | 386          | 180      | 46.63%        |
| 2023 | 378          | 102      | 26.98%        |

---

## 📁 Project Structure

```
AI_Review/
│
├── download_review.ipynb       # Step 1: Download ICLR data
├── generate_review.ipynb       # Step 2: Generate AI reviews
├── finetune_lm.ipynb          # Step 3: Train classifier
├── inference.ipynb            # Step 4: Run predictions
│
└── data/ (in Google Drive)
    ├── iclr_2020_data/
    │   ├── pdfs/              # Paper PDFs
    │   └── iclr_2020_reviews.csv
    ├── iclr_2021_data/
    ├── iclr_2022_data/
    ├── iclr_2023_data/
    │
    ├── ai_review_2020.csv     # Generated reviews (training data)
    │
    ├── finetuned_longformer_lora/  # Trained model
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── tokenizer files
    │
    └── inference_results/     # Prediction outputs
```

## 🤖 Model Details

### Base Model
- **Name**: Longformer Base (4096 tokens)
- **Parameters**: 149M total
- **Architecture**: Transformer with efficient attention mechanism
- **Context Length**: 4096 tokens (vs 512 for BERT)

### Fine-tuning Strategy
- **Method**: LoRA (Low-Rank Adaptation)
- **Trainable Parameters**: 887K (0.59% of total)
- **Target Modules**: Query and Value attention matrices
- **Rank**: 8
- **Alpha**: 16

### Training Data
- **Real Reviews**: ICLR 2020 official reviews (label 0)
- **AI Reviews**: GPT-4o/DeepSeek generated reviews (label 1)
- **Balance**: Equal samples per class (~100-300 each)
- **Split**: 80% training, 20% validation

### Inference Performance
- **Speed**: ~1.2 seconds per batch (8 reviews)
- **Memory**: ~6GB GPU memory
- **Throughput**: ~6-7 reviews/second on NVIDIA L4

## 📊 Results

The model successfully identifies patterns that distinguish AI-generated reviews from human-written ones:

### Key Findings

1. **AI Detection Capability**: The classifier achieves strong performance in distinguishing between real and synthetic reviews

2. **Temporal Trends**: Analysis shows varying levels of AI-generated content across years:
   - 2021: Higher proportion of reviews flagged as AI-like (68%)
   - 2022: Moderate proportion (47%)
   - 2023: Lower proportion (27%)

3. **Review Characteristics**: The model learns subtle differences in:
   - Writing style and structure
   - Technical depth and specificity
   - Critique patterns and feedback style
   - Language consistency and fluency

### Limitations

- **Training Data**: Model trained on specific AI models (GPT-4o, DeepSeek)
- **Generalization**: Performance may vary with different AI models
- **Review Length**: Optimized for typical review lengths (200-1600 tokens)
- **Domain**: Focused on ML/AI conference reviews (ICLR)

## 🔧 Customization

### Using Different Data Sources

Modify the data paths in each notebook to use different conferences or journals:

```python
# Example: Using NeurIPS instead of ICLR
venue_id = f'NeurIPS.cc/{year}'
```

### Adjusting Model Parameters

In `finetune_lm.ipynb`, experiment with:
- LoRA rank (4, 8, 16) - higher = more capacity but more parameters
- Learning rate (1e-4 to 5e-4)
- Batch size (4, 8, 16) - depends on GPU memory
- Max length (1024, 2048, 4096) - based on review lengths

### Using Alternative Models

Replace Longformer with other models:
- `roberta-base` for shorter texts
- `bigbird-roberta-base` for very long documents
- `deberta-v3-base` for improved performance

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@misc{ai_review_detection_2025,
  title={AI Review Detection: A Pipeline for Identifying AI-Generated Academic Reviews},
  author={Your Name},
  year={2025},
  url={https://github.com/sy-shen/AI_Review}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests.

## 📄 License

This project is provided for research and educational purposes.

## 🙏 Acknowledgments

- **OpenReview**: For providing access to ICLR review data
- **Hugging Face**: For the Transformers library and pre-trained models
- **Microsoft Research**: For the LoRA fine-tuning technique
- **AllenAI**: For the Longformer model

## 📧 Contact

For questions or collaborations, please open an issue on GitHub or contact through the repository.

---

**Note**: This project is designed for Google Colab but can be adapted for local execution with appropriate GPU resources. Make sure to configure API keys and file paths according to your environment.

