# AI Review Detection: ICLR Paper Review Analysis

This project implements a complete pipeline for detecting AI-generated paper reviews using fine-tuned language models. The system downloads ICLR conference data, generates synthetic reviews using AI models, trains a classifier to distinguish between real and AI-generated reviews, and performs inference on real data.

## Overview

This project addresses the growing concern of AI-generated academic reviews by:

1. **Collecting real reviews** from ICLR (International Conference on Learning Representations)
2. **Generating synthetic reviews** using LLMs (GPT-4o, DeepSeek Reasoner)
3. **Training a classifier** using Longformer with LoRA fine-tuning to detect AI-generated content
   - **Training Set**: Uses ICLR 2021 reviews (both real human-written reviews and AI-generated synthetic reviews)
   - **Test Set**: Performs inference on real reviews from ICLR 2022-2025 to detect potential AI-generated content and analyze trends across multiple years

## Pipeline Workflow

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

## Results

### Model Performance on Training Set (ICLR 2021)

The fine-tuned Longformer+LoRA model achieves excellent performance on the 2021 training set:

**Confusion Matrix:**

|                    | Predicted Real | Predicted AI |
|--------------------|----------------|--------------|
| **Actual Real**    | 160            | 0            |
| **Actual AI**      | 0              | 160          |

**Classification Report:**

| Class              | Precision | Recall | F1-Score | Support |
|--------------------|-----------|--------|----------|---------|
| Real Review (0)    | 1.00      | 1.00   | 1.00     | 160     |
| AI Generated (1)   | 1.00      | 1.00   | 1.00     | 160     |
| **Accuracy**       |           |        | 1.00     | 320     |
| **Macro Avg**      | 1.00      | 1.00   | 1.00     | 320     |
| **Weighted Avg**   | 1.00      | 1.00   | 1.00     | 320     |


### AI Detection Trends (2022-2025)

Our fine-tuned classifier detected a significant increasing trend in potential AI-generated reviews across ICLR conferences from 2022 to 2025:

![Percentage of Real Reviews Classified as AI-Generated](images/ai_percentage_trend.png)

### Summary Statistics

| Year | Total Reviews | AI-Detected | Percentage |
|------|--------------|-------------|------------|
| 2022 | 1,937        | 2           | 0.10%      |
| 2023 | 1,887        | 0           | 0.00%      |
| 2024 | 1,818        | 133         | 7.32%      |
| 2025 | 1,961        | 392         | 19.99%     |


## Discussion
- Each year, we randomly sampled reviews from approximately 500 papers, resulting in around 2,000 review entries annually.
- In 2022 and 2023, the classifier detected virtually no AI-generated reviews. This is likely because ChatGPT and similar models were only released after the 2023 ICLR review process, so AI-generated content was not present in the earlier years; this further demonstrates the robustness and specificity of our model.
- Starting from ICLR 2024 and continuing into 2025, we observe a marked increase in the proportion of reviews classified as AI-generated, suggesting a rising trend in the potential usage of AI tools in review writing.



