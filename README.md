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
- [Baselines](#-baselines)
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
Our STA-GANN and STKriging implementations are developed based on **BasicTS (Dec 2024 release)**.  
We plan to continuously follow the updates of BasicTS to ensure that our framework remains fully aligned with the official version.  

The current repository structure is as follows:
```bash
├── data_preparation/   # Dataset processing methods
├── datasets/           # Raw data
├── examples/           # Parameters CFG for each dataset and each methods
├── ── {Method}/        # Corresponding method folder
├── ── ── {Method}_{Datasets}.py # Each method and dataset has a CFG py file.
├── stkriging/         # Main Folder
├── ── arch/           # Kriging algorithm
├── ── data/           # Dataset processing related
├── ── loss/           # Define the loss function, redirected from metrics
├── ── metrics/        # metrics
├── ── runners/        # pipeline
├── ── utils/          # Kriging processing related
└── README.md          # Project documentation
```
## 🚀 Getting Started

Please install all dependencies via:
```bash
pip install -r requirements.txt
```
Go to examples/run.py, select the method and dataset you need, then run:
```bash
python examples/run.py
```
---
## 📊 Datasets

| Dataset   | Domain     | Sensors | Duration            | Time Steps | Frequency | Extra Info | Data Link |
|-----------|------------|---------|---------------------|------------|-----------|------------|-----------|
| METR-LA | Traffic    | 207     | Mar 2012 – Jun 2012 | 34,272     | 5 min     | Includes latitude/longitude; sensor graph; adjacency via Gaussian kernel | [LINK](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY | Traffic    | 325     | Jan 2017 – Jun 2017 | 52,116     | 5 min     | Includes latitude/longitude; sensor graph; adjacency via Gaussian kernel | [LINK](https://github.com/liyaguang/DCRNN) |
| PEMS03 | Traffic    | 358     | Sep 2018 – Nov 2018 | 26,208     | 5 min     | Sensor graph only; no latitude/longitude | [LINK](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS04 | Traffic    | 307     | Jan 2018 – Feb 2018 | 16,992     | 5 min     | Sensor graph only; no latitude/longitude | [LINK](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS07 | Traffic    | 883     | May 2017 – Aug 2017 | 28,224     | 5 min     | Sensor graph only; no latitude/longitude | [LINK](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| PEMS08 | Traffic    | 170     | Jul 2016 – Aug 2016 | 17,856     | 5 min     | Sensor graph only; no latitude/longitude | [LINK](https://github.com/guoshnBJTU/ASTGNN/tree/main/data) |
| NREL | Energy     | 137     | 2006                | 105,120    | 10 min\*  | Solar power plants in Alabama; includes latitude/longitude | [LINK](https://github.com/Kaimaoge/IGNNK) |
| USHCN | Climate    | 1,218   | 1899 – 2019         | 1,440      | Monthly   | Precipitation; includes latitude/longitude | [LINK](https://github.com/Kaimaoge/IGNNK) |
| AQI | Environment| 437     | 43 cities (China)   | 59,710     | Hourly\*  | Air Quality Index (PM2.5); includes latitude/longitude | [LINK](https://github.com/Graph-Machine-Learning-Group/grin) |

---
## 📚 Baselines

| Name     | Paper Title                                                                 | Venue   | Year | Link                                                                                  | Type |
|----------|-----------------------------------------------------------------------------|---------|------|---------------------------------------------------------------------------------------|------|
| GCN      | Semi-Supervised Classification with Graph Convolutional Networks            | ICLR    | 2017 | [LINK](https://arxiv.org/abs/1609.02907)                                              | Backbone |
| GIN      | How Powerful are Graph Neural Networks?                                     | ICLR    | 2019 | [LINK](https://arxiv.org/abs/1810.00826)                                              | Backbone |
| IGNNK    | Inductive Graph Neural Networks for Spatiotemporal Kriging                  | AAAI    | 2021 | [LINK](https://arxiv.org/abs/2006.07527)                                              | Spatio-temporal Kriging |
| GRIN     | Filling the Gaps: Multivariate Time Series Imputation by Graph Neural Networks | ICLR    | 2022 | [LINK](https://arxiv.org/abs/2108.00298)                                              | Adapted Spatio-temporal Imputation |
| SATCN    | Spatial Aggregation and Temporal Convolution Networks for Real-time Kriging | None    | 2021 | [LINK](https://arxiv.org/pdf/2109.12144.pdf)                                          | Spatio-temporal Kriging |
| INCREASE | INCREASE: Inductive Graph Representation Learning for Spatio-Temporal Kriging | WWW     | 2023 | [LINK](https://arxiv.org/abs/2302.02738)                                              | Spatio-temporal Kriging |
| DualSTN  | Decoupling Long- and Short-Term Patterns in Spatiotemporal Inference        | TNNLS   | 2023 | [LINK](https://arxiv.org/pdf/2109.09506v3)                                            | Spatio-temporal Kriging |
| IAGCN    | Inductive and Adaptive Graph Convolution Networks Equipped with Constraint Task for Spatial–Temporal Traffic Data Kriging | KBS     | 2024 | [LINK](https://www.sciencedirect.com/science/article/abs/pii/S0950705123010730)       | Spatio-temporal Kriging |
| OKriging | —                                                                           | —       | —    | —                                                                                     | Traditional Kriging |

---
## 📜 Citation

arXiv link & citation is coming soon!

---

## 🤝 Contact

For any issues, please contact: `liyujie23s@ict.ac.cn`

