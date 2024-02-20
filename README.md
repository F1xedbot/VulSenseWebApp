# VulSenseWebApp

## Description
VulSenseWebApp is a full-stack implementation of VulnSense, a leading multimodal machine learning model for smart contract vulnerability detection on Ethereum.

VulnSense enhances detection efficiency through its novel multimodal deep learning architecture. It comprehensively analyzes both the source code and documentation of smart contracts. Specifically, VulnSense leverages BERT, Bi-LSTM, and GNN models to extract semantic and syntactic features from code, text and bytecode.

These multisource features are then fused and classified through fully connected layers to predict potential vulnerabilities. By incorporating information from code, text and program structure, VulnSense significantly outperforms previous single-model approaches in vulnerability identification accuracy.

## ML Model Folder
Access the ML model folder [here](https://drive.google.com/file/d/1T7NiqtR3VcSOanwi8P7K0fI2Ui8kcnlL/view?usp=sharing).

## How to Use
### 1. Extraction Module
Three modules are utilized in this project:
   - Bytecode Embedding using GNN
   - Source Code Embedding using BERT
   - Opcode Embedding using Bi-LSTM

### 2. Model
Download the provided zip file, extract it, and place it in the project folder. Upon extraction, you will find three subfolders (M1, M2, M3) representing models built from different combinations of extraction modules. The "VulnSense" folder contains the model utilizing all three modules for its final embedding.

### 3. Frameworks
Two frameworks are available for building the web app:
   - Execute `scaffold.py` to run the web app using the Streamlit framework. This option provides a fast and resource-efficient solution, although the UI may exhibit occasional unreliability.
   - Execute `app.py` to run the web app using the Flask framework. This option ensures a reliable and stable performance, albeit with longer loading times and higher resource usage.

## Demo (Streamlit)

*Link to Streamlit demo.*

This project aims to enhance smart contract security by leveraging advanced deep learning techniques. Thank you for your interest and support!
