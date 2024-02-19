from flask import Flask, render_template, request
from flask_caching import Cache

from threading import Thread
from time import sleep
import json
import re
import textwrap
from base import getVersionPragma, clean_opcode, remove_comment, get_bytecode, return_bytecode_opcode, bytecode_to_cfg
import plotly.graph_objects as go
import requests
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import plotly.express as px

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

import networkx as nx
import tensorflow as tf
import tensorflow_text as text
from graphviz import Source

from PIL import Image, ImageOps

from torch_geometric.data import Data
from collections import Counter
from io import BytesIO 

import torch
from GnnDataset import make_jraph_dataset
from GnnModel import GCN

from tensorflow.keras.preprocessing.text import Tokenizer   
from tensorflow.keras.preprocessing.sequence import pad_sequences

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# tell Flask to use the above defined config
app.config.from_mapping(config)
cache = Cache(app)


progress_status = None
uploaded_content = None
flow_data = None
results = None
freeze_time = 0.4

API_KEY = 'QEAPNG89EK8R3K1EXAM4HZCKESPIDGNQQH'
BASE_URL = 'https://api.etherscan.io/api'

def predict(input_data, gnn_input, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN):
    global progress_status
    global results
    VOCAB_OP_SIZE = input_data['opcode'].nunique()
    MAX_OPCODE_LENGTH = 200
    EMBEDDING_SIZE = 256
    time.sleep(freeze_time)
    progress_status = 75
    time.sleep(freeze_time)
    
    # Khởi tạo tokenizer
    tokenizer = Tokenizer(num_words=1411)

    # Fit tokenizer với các opcode
    tokenizer.fit_on_texts(input_data['opcode'])
    progress_status = 80
    # Assuming input_data is a dataframe or similar structure with 'opcode' and 'source_code' columns
    sequences = tokenizer.texts_to_sequences(input_data['opcode'])
    opcode_matrix = pad_sequences(sequences, maxlen=MAX_OPCODE_LENGTH)
    
    
    def predict_model(model, inputs, name):
        print(f"Getting {name} Prediction...")
        start_time = time.time()
        y_pred = model.predict(inputs)
        y_pred_classes = np.argmax(y_pred, axis=1)
        elapsed_time = time.time() - start_time
        print(f"{name} Prediction Time: {elapsed_time:.2f} seconds")
        return y_pred_classes, elapsed_time

    progress_status = 85
    y_pred_classes_m1, time_m1 = predict_model(BERT_BiLSTM, [input_data['source_code'], opcode_matrix], "BERT_BiLSTM")
    time.sleep(2)
    progress_status = 90
    y_pred_classes_m2, time_m2 = predict_model(BERT_GNN, [input_data['source_code'], gnn_input], "BERT_GNN")
    time.sleep(2)
    progress_status = 95
    y_pred_classes_m3, time_m3 = predict_model(BiLSTM_GNN, [opcode_matrix, gnn_input], "BiLSTM_GNN")
    time.sleep(2)
    progress_status = 98
    y_pred_classes_vulnsense, time_vulnsense = predict_model(VulnSense, [input_data['source_code'], opcode_matrix, gnn_input], "VulnSense")
    time.sleep(freeze_time)
    progress_status = 100
    results = [[y_pred_classes_m1[0], y_pred_classes_m2[0], y_pred_classes_m3[0], y_pred_classes_vulnsense[0]], [time_m1, time_m2, time_m3, time_vulnsense]]


def preprocess(input_data):
    global progress_status
    global flow_data
    progress_status = 5
    time.sleep(freeze_time)
    progress_status = 16
    time.sleep(freeze_time)
    source_code = remove_comment(input_data)
    progress_status = 20
    time.sleep(freeze_time)
    progress_status = 25
    time.sleep(freeze_time)
    bytecode, opcode = return_bytecode_opcode(source_code)
    progress_status = 32
    time.sleep(freeze_time)
    progress_status = 34
    time.sleep(freeze_time)
    lines, nodes, edges = bytecode_to_cfg(bytecode)
    dataset = make_jraph_dataset(nodes, edges)
    time.sleep(freeze_time)
    progress_status = 42
    nodes = torch.tensor(dataset[0]['input_graph'][0], dtype=torch.float)
    edges = torch.tensor(dataset[0]['input_graph'][1], dtype=torch.float)
    senders = torch.tensor(dataset[0]['input_graph'][3], dtype=torch.long)
    receivers = torch.tensor(dataset[0]['input_graph'][2], dtype=torch.long)
    label = torch.tensor([0], dtype=torch.long)
    edge_index = torch.stack([senders, receivers], dim=0)
    data = Data(edge_index=edge_index, x=nodes, edge_attr=edges, y=label)
    time.sleep(freeze_time)
    progress_status = 50

    data_list = []
    data_list.append(data)

    loader = DataLoader(data_list, batch_size=32, shuffle=False)


    model = GCN(hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        for data in loader:  # Iterate in batches over the training dataset.
                out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
                loss = criterion(out, data.y)  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                optimizer.zero_grad()  # Clear gradients.
                return(out.detach().tolist()[0])
    time.sleep(freeze_time)
    progress_status = 60
    gnn_input = train()
    for i in range(len(gnn_input)):
        gnn_input[i] = gnn_input[i]
    time.sleep(freeze_time)
    progress_status = 65
    gnn_input = np.array([[gnn_input]])
    time.sleep(freeze_time)
    progress_status = 70
    print(gnn_input)
    global VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN
    flow_data = [input_data, lines, data]
    predict(pd.DataFrame({'opcode': [opcode], 'source_code': [source_code]}), gnn_input, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN)

VulnSense = None
BERT_BiLSTM = None
BERT_GNN = None
BiLSTM_GNN = None

def load_model():
    VulnSense = tf.keras.models.load_model('Model/Model/VulnSense')
    print("Loaded VulnSense")
    BERT_BiLSTM = tf.keras.models.load_model('Model/Model/M1')
    print("Loaded BERT_BiLSTM")
    BERT_GNN = tf.keras.models.load_model('Model/Model/M2')
    print("Loaded BERT_GNN")
    BiLSTM_GNN = tf.keras.models.load_model('Model/Model/M3')
    print("Loaded BiLSTM_GNN")
    return VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN

@app.route('/')
@cache.cached(timeout=1000)
def index():
  global VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN
  VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN = load_model()
  return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_content  # Access the global variable
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Save the file to the system
    file.save(file_path)

    # Read the content of the file
    with open(file_path, "r") as code_file:
        uploaded_content = code_file.read()
        # Perform further operations with the file content if needed (e.g., analysis)

    # Delete the temporary file
    os.remove(file_path)
    
    print(VulnSense)

    return 'File uploaded successfully'

@app.route('/analyze', methods=['GET'])
def analyze():
    global uploaded_content
    if uploaded_content is None:
        return 'No file uploaded'

    preprocess_thread = Thread(target=preprocess, args=(uploaded_content,))
    preprocess_thread.start()
    
    return 'Scan Completed'

@app.route('/progress', methods=['GET'])
def getProgress():
  statusList = {'status':progress_status}
  return json.dumps(statusList)
@app.route('/show-analysis')
def getResult():
    global results
    global flow_data
    
    def calculate_score(predictions):
        security_scores = {
            1: 0,
            0: 35,
            2: 25 
        }
        score = 0
        count = 0
        for prediction in predictions:
            score += security_scores[prediction]
            if prediction != 1:
                count += 1
        
        return score, count
    
    def progress_score(progress):
        text = None
        color = None
        if progress > 60:
            text = "Your contract is at critical security risk "
            color = "red"
        elif progress > 20:
            text = "Your contract is dangerous but not at risk"
            color = "orange"
        else:
            text = "Your contract is safe"
            color = "green"
        return text, color
    
    progress_value, flag_count = calculate_score(results[0])
    text, color = progress_score(progress_value)
    
    severity_counts = {
        'high': 0,
        'medium': 0,
        'critical': 0
    }
    for vul in results[0]:
        if vul == 0:
            severity_counts['high'] += 1
        if vul == 2:
            severity_counts['medium'] += 1
                
    unique_non_one_values = [num for num, count in Counter(results[0]).items() if num != 1 and count > 0]
    codelines = len(flow_data[0].splitlines())
    
    print(severity_counts)
    print(len(unique_non_one_values))
    
    prediction_text_format = {
        1: '✅ Undetected',
        2: '⚠️ Reentrancy',
        0: '⭕ Arithmetic'
    }
    
    elapsed_time = sum([sec for sec in results[1]]) + 6
    return render_template('analysis.html', progress_value=f'{progress_value:.2f}', text=text, color=color, severity_counts=severity_counts, flag_count=flag_count, 
                           unique_non_one_values=len(unique_non_one_values), elapsed_time=f'{elapsed_time:.2f}', codelines=codelines,
                           results=results, prediction_text_format=prediction_text_format)

if __name__ == '__main__':
    app.run(debug=True)