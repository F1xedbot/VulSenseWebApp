# VulSenseWebApp
An implementation of VulnSense-smart contract vulnerability detection framework (source code &amp; demo)


A multimodal deep learning framework for efficient smart contract vulnerability detection. VulnSense analyzes both the source code and documentation of Ethereum smart contracts. It uses BERT and Bi-LSTM models to extract features from both the code and text. These features are then fed into a fully connected layer to predict vulnerabilities. VulnSense leverages a multimodal architecture to achieve superior detection accuracy compared to prior single-model approaches.

ML Model folder : https://drive.google.com/file/d/1T7NiqtR3VcSOanwi8P7K0fI2Ui8kcnlL/view?usp=sharing

How to use:
- Extraction Module: 3 modules being used in this project:
  + Bytecode Embedding using GNN
  + Source Code Embedding using BERT
  + Opcode Embedding using Bi-LSTM

- Model: There is a link to a zip file, extract it and place it inside the same folder as the project, upon extracting, you will see it contains 3 subfolders of 4 models used to build the application, M1, M2, M3, and VulnSense, M1 -> M3 is the model that is built from the different combinations of extraction modules and VulnSense is the model that using all the modules for its final embedding

- There are 2 different frameworks that you can use to build the web app in this project
  + You can set up and run the file 


