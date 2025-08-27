# 🧬 Hyperspectral Image Segmentation of Head & Neck Squamous Cell Carcinoma  

[![GitHub repo](https://img.shields.io/badge/GitHub-Project-green?logo=github)](https://github.com/vinaykadam007/Hyper-spectral-project)  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)  
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-orange)]()  
[![HPC](https://img.shields.io/badge/High%20Performance%20Computing-HPC-lightgrey)]()  

---

## 📌 Project Overview  
This project focuses on **segmentation of hyperspectral images (HSI)** of **head and neck squamous cell carcinoma (HNSCC)** surgical specimens. Accurate cancer margin detection during surgery is critical to ensure complete tumor removal and improve patient outcomes.  

We designed and implemented a **Wavelet Convolutional Neural Network (Wavelet CNN)**, trained on **hyperspectral imaging data from 100+ patient surgical specimens**, leveraging both **spectral** and **spatial features**. The model was deployed and trained on an **HPC cluster** to handle large-scale, memory-intensive datasets.  

---

## 🎯 Objective & Outcome  

- **Objective:**  
  To develop a deep learning pipeline capable of segmenting hyperspectral surgical specimen images of HNSCC, enabling accurate tumor margin detection for clinical use.  

- **Outcome:**  
  - Implemented a **Wavelet CNN** trained on **7,000+ hyperspectral patches** from 100+ patients.  
  - Optimized **memory management and preprocessing pipelines** for HPC deployment.  
  - Achieved **0.74 IoU** on unseen test data across **three segmentation classes (tumor, normal, tumor-involved)**.  
  - Demonstrated a scalable approach with potential to aid **real-time intraoperative margin detection**.  

---

## ⚙️ Methodology  

- **Data Source:** Hyperspectral imaging data of HNSCC surgical specimens, collected at Emory University Hospital.  
- **Preprocessing:**  
  - Data split into tumor (T), normal (N), and tumor-involved (TN) samples.  
  - Image patches extracted (24×24×91 spectral bands).  
  - Standardization + spectral band normalization.  

- **Model:** Wavelet CNN  
  - Combines **wavelet transforms** (spectral features) + **2D CNN layers** (spatial features).  
  - Dense connections, Haar wavelets, and channel-wise concatenation for efficient feature extraction.  
  - Trained on **NVIDIA Titan-XP GPUs** with Adam optimizer and early stopping.  

- **Evaluation Metric:** Intersection over Union (IoU).  

---

## 📊 Results  

| Experiment                     | IoU Score | Notes |
|--------------------------------|-----------|-------|
| Wavelet CNN + Factor Analysis   | **0.48**  | Faster training but poor separation (loss of spectral info). |
| Wavelet CNN (full spectral)     | **0.59**  | Improved separation between tumor vs. normal. |
| Optimized Wavelet CNN (HPC run) | **0.74**  | Strongest results on unseen test specimens. |  

**Key Insight:** Preserving all **91 spectral bands** leads to better feature extraction and segmentation accuracy compared to dimensionality reduction.  

---

## 📂 Repository Structure  

```
Hyper-spectral-project/
│── data/                 # Sample datasets or links
│── notebooks/            # Jupyter notebooks for preprocessing & training
│── src/                  # Core training scripts and Wavelet CNN model
│── results/              # Outputs, visualizations, IoU metrics
│── requirements.txt      # Dependencies
│── README.md             # Project documentation
```

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.8+  
- PyTorch / TensorFlow (depending on implementation)  
- NumPy, OpenCV, Matplotlib  

Install dependencies:  
```bash
pip install -r requirements.txt
```

### Run Training  
```bash
python src/train.py --config configs/hsi_config.yaml
```

### Run Inference  
```bash
python src/inference.py --image_path sample_data/hsi_sample.png
```

---

## 📌 Future Work  

- Expand dataset to include more surgical specimens for better generalization.  
- Explore hybrid architectures (e.g., ResNet + Wavelet CNN).  
- Investigate real-time integration for **intraoperative margin detection**.  

---

## 🙌 Acknowledgements  
This work was conducted at the **Quantitative Bioimaging Lab (QBIL), UT Dallas**, with support and guidance from **Dr. Baowei Fei**.  

---

✨ If you find this project useful, please ⭐ the repo and cite our work!
