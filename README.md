# AI Review Detection: ICLR Paper Review Analysis

This project implements a complete pipeline for detecting AI-generated paper reviews using fine-tuned language models. The system downloads ICLR conference data, generates synthetic reviews using AI models, trains a classifier to distinguish between real and AI-generated reviews, and performs inference on new data.

## Overview

This project addresses the growing concern of AI-generated academic reviews by:

1. **Collecting real reviews** from ICLR (International Conference on Learning Representations)
2. **Generating synthetic reviews** using state-of-the-art LLMs (GPT-4o, DeepSeek Reasoner)
3. **Training a classifier** using Longformer with LoRA fine-tuning to detect AI-generated content
4. **Analyzing trends** in AI-generated reviews across multiple years (2021-2023)

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
