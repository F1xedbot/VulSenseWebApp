# VulSenseWebApp

## Description
VulSenseWebApp is an implementation of the VulnSense smart contract vulnerability detection framework. This innovative framework employs a multimodal deep learning approach to enhance the efficiency of smart contract vulnerability detection on the Ethereum blockchain. VulnSense comprehensively analyzes both the source code and documentation of Ethereum smart contracts, utilizing BERT and Bi-LSTM models to extract features from code and text. The extracted features are then processed through a fully connected layer to predict potential vulnerabilities. The use of a multimodal architecture distinguishes VulnSense by achieving superior detection accuracy compared to previous single-model approaches.

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
