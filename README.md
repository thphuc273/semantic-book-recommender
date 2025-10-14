---
title: Sentiment Analysis Dashboard
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "4.39.0"
app_file: app.py
pinned: true
---

# Sentiment Analysis Dashboard 📊

This project is an interactive Gradio web app for analyzing **book emotions** based on their descriptions.  
It combines NLP and emotion classification models to visualize the emotional distribution of books.

## 🚀 Features
- Upload or search books with emotion detection
- Supports emotions: joy, sadness, anger, fear, disgust, surprise, and neutral
- Visual charts and comparison between predicted vs. labeled emotions
- Built with `transformers`, `pandas`, and `gradio`

## 🧠 Model
Uses a pre-trained Hugging Face emotion classifier fine-tuned on social media text.

## ⚙️ Deployment
CI/CD is automated via **GitHub Actions**, deploying directly to Hugging Face Spaces.
