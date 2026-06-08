# ArcFace Region Sensitivity Analysis using InsightFace

## Overview
This project explores **region-wise sensitivity in face recognition models** using ArcFace embeddings from InsightFace.  
The goal is to analyze how different facial regions (eyes, nose, mouth, upper face, lower face) contribute to identity recognition performance.

---

## Motivation
Modern face recognition models like ArcFace generate embeddings for identity matching.  
However, different facial regions contribute unequally to recognition performance.

This project investigates **which facial regions carry the most discriminative identity information** and how masking affects recognition accuracy.

---

## Key Features
- Face detection using InsightFace
- Region-based face masking (eyes, nose, mouth, upper, lower face)
- Embedding extraction for full face and individual regions
- Cosine similarity comparison between embeddings
- Visualization of region sensitivity results

---

## Methodology
1. Detect face using InsightFace
2. Split face into different regions
3. Mask each region separately
4. Extract ArcFace embeddings for:
    - Full face
    - Each masked region
5. Compute cosine similarity between embeddings
6. Analyze which regions contribute most to identity recognition

---

## Project Structure
arcface_region_sensitivity_insightface/
│
├── src/
│ ├── face_detection.py
│ ├── embedding.py
│ ├── region_masking.py
│ └── analysis.py
│
├── data/
│ ├── raw_images/
│ └── masked_images/
│
├── results/
│ └── plots/
│
├── main.py
├── README.md
└── requirements.txt


---

## Tech Stack
- Python
- InsightFace (ArcFace)
- OpenCV
- NumPy
- Matplotlib

---

## Results
- Eye and central face regions show highest contribution to identity recognition
- Lower face regions show lower discriminative power
- Region masking significantly reduces embedding similarity

(Add your plots from `results/plots/` here)

---

## How to Run
```bash
pip install -r requirements.txt
python main.py

