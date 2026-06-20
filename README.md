# 🌍 AfriVoices Edge Multilingual ASR Engine

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

An optimized, lightweight, single unified Automatic Speech Recognition (ASR) engine built for the **AfriVoices East Africa ASR Hackathon 2026**. This system is specifically architected to deliver high-fidelity transcriptions across six low-resource East African languages while strictly adhering to edge-compute hardware constraints.

## 📋 Project Objective & Scope

Most modern speech technologies overlook African linguistic depth, creating an intense digital divide. This project fine-tunes a unified transformer-based ASR architecture capable of recognizing speech across:
* **Swahili** (`swa`) — 100% Spontaneous conversational data.
* **Kikuyu (Gĩkũyũ)** (`kik`) — Scripted and spontaneous variations.
* **Luo (Dholuo)** (`luo`) — Multi-dialectal speech samples.
* **Somali** (`som`) — Regional conversational structures.
* **Kalenjin** (`kal`) — Nandi & Kipsigis dialects.
* **Maasai** (`mas`) — Kimasasi & Kioamburu dialects.

The core engineering focus centers on balancing the stark **dataset size imbalance** (~3,000 hours of Swahili vs. ~500 hours of Maasai) to lower the **Unweighted Mean Word Error Rate (WER)** across all target languages simultaneously.

---

## ⚡ Strict Edge-Hardware Guarantees

To prevent disqualification and guarantee seamless real-world deployment on mobile devices and low-cost hardware (e.g., Raspberry Pi 4), the architecture enforces the following hard boundaries:
* **Parameter Threshold:** Strictly `< 1 Billion parameters` total.
* **Memory Footprint:** Strictly `≤ 8 GB RAM` peak utilization during active inference.
* **Compute Target:** Optimized for **CPU-only execution environments** (No GPU required post-training).
* **Latency Ceiling:** Real-time or near real-time transcription speeds (`≤ 2x audio duration` processing speed on CPU).
* **Connectivity:** 100% Offline operational capabilities.

---

## 🛠️ Repository Architecture

Our project emphasizes production-grade modularity rather than monolithic notebooks, ensuring clean pipeline execution:

```text
AfriVoices/
│
├── data/                  # Local cache directory (Git ignored)
│
├── src/                   # Source Core Engine
│   ├── __init__.py
│   ├── data_loader.py     # Live HF dataset streaming handling key discrepancies
│   ├── preprocessing.py   # 16kHz audio resampling & diacritic-safe text cleaners
│   ├── train.py           # Balanced interleaving loops and training execution
│   └── utils.py           # Evaluation hooks (WER tracking via jiwer)
│
├── notebooks/             # Exploratory cloud training workspaces
│   └── baseline_exploration.ipynb
│
├── .gitignore             # Binary and large cache filters
├── LICENSE                # Open-source licensing compliance (Apache 2.0)
└── README.md              # Project documentation master