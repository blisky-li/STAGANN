# STA-GANN: A Valid and Generalizable Spatio-Temporal Kriging Approach

Official repository for the paper:  
**"STA-GANN: A Valid and Generalizable Spatio-Temporal Kriging Approach"**  
Accepted at **ACM CIKM 2025**.  

---

## 📖 Table of Contents
- [Introduction](#-introduction)
- [Key Contributions](#-key-contributions)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
- [Datasets](#-datasets)
- [Results](#-results)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 📖 Introduction

Spatio-temporal tasks often encounter incomplete data due to missing or inaccessible sensors, making **spatio-temporal kriging** crucial for inferring missing temporal information.  

However, existing models face challenges in:  
- Capturing **dynamic spatial dependencies** and **temporal shifts**,  
- Ensuring **validity** of spatio-temporal patterns,  
- Optimizing **generalizability** to unknown sensors.  

To address these issues, we propose **Spatio-Temporal Aware Graph Adversarial Neural Network (STA-GANN)**, a novel GNN-based kriging framework that enhances both validity and generalization of spatio-temporal pattern inference.  

---

## 🔑 Key Contributions
- **Decoupled Phase Module (DPM):** Detects and adjusts timestamp shifts.  
- **Dynamic Data-Driven Metadata Graph Modeling (DMGM):** Updates spatial relationships using temporal signals and metadata.  
- **Adversarial Transfer Learning Strategy:** Ensures robust generalization to unseen sensors.  

---

## 📂 Repository Structure
```bash
├── data/              # Datasets (or instructions to download)
├── models/            # Core model implementations
├── configs/           # YAML config files
├── utils/             # Helper functions
├── train.py           # Training script
├── test.py            # Evaluation script
└── README.md          # Project documentation

