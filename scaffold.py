import streamlit as st
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

freeze_time = 0.4

API_KEY = 'QEAPNG89EK8R3K1EXAM4HZCKESPIDGNQQH'
BASE_URL = 'https://api.etherscan.io/api'


def preprocess(input_data, scanning_progress):
    scanning_progress.progress(5, text=f'{5}% | Creating dataframe...')
    time.sleep(freeze_time)
    scanning_progress.progress(16, text=f'{16}% | Adding source code...')
    time.sleep(freeze_time)
    source_code = remove_comment(input_data)
    scanning_progress.progress(20, text=f'{20}% | Done cleaning source code!')
    time.sleep(freeze_time)
    scanning_progress.progress(25, text=f'{25}% | Adding bytecode and opcode...')
    time.sleep(freeze_time)
    bytecode, opcode = return_bytecode_opcode(source_code)
    scanning_progress.progress(32, text=f'{32}% | Done extracting bytecode and opcode from source!')
    time.sleep(freeze_time)
    
    scanning_progress.progress(34, text=f'{34}% | Converting bytecode to graph...')
    time.sleep(freeze_time)
    lines, nodes, edges = bytecode_to_cfg(bytecode)
    dataset = make_jraph_dataset(nodes, edges)
    time.sleep(freeze_time)
    scanning_progress.progress(42, text=f'{42}% | Extracting embedding from nodes...')
    nodes = torch.tensor(dataset[0]['input_graph'][0], dtype=torch.float)
    edges = torch.tensor(dataset[0]['input_graph'][1], dtype=torch.float)
    senders = torch.tensor(dataset[0]['input_graph'][3], dtype=torch.long)
    receivers = torch.tensor(dataset[0]['input_graph'][2], dtype=torch.long)
    label = torch.tensor([0], dtype=torch.long)
    edge_index = torch.stack([senders, receivers], dim=0)
    data = Data(edge_index=edge_index, x=nodes, edge_attr=edges, y=label)
    time.sleep(freeze_time)
    scanning_progress.progress(50, text=f'{50}% | Creating graph dataset...')

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
    scanning_progress.progress(60, text=f'{60}% | Re-training GNN...')
    gnn_input = train()
    for i in range(len(gnn_input)):
        gnn_input[i] = gnn_input[i]
    time.sleep(freeze_time)
    scanning_progress.progress(65, text=f'{65}% | Getting GNN ouputs')
    gnn_input = np.array([[gnn_input]])
    time.sleep(freeze_time)
    scanning_progress.progress(70, text=f'{70}% | Done creating Dataframe!')
    print(gnn_input)
    return pd.DataFrame({'opcode': [opcode], 'source_code': [source_code]}), gnn_input, [lines, data]

def predict(input_data, gnn_input, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN, scanning_progress):
    
    VOCAB_OP_SIZE = input_data['opcode'].nunique()
    MAX_OPCODE_LENGTH = 200
    EMBEDDING_SIZE = 256
    time.sleep(freeze_time)
    scanning_progress.progress(75, text=f'{75}% | Initializing Text Tokenizer...')
    time.sleep(freeze_time)
    
    # Khởi tạo tokenizer
    tokenizer = Tokenizer(num_words=1411)

    # Fit tokenizer với các opcode
    tokenizer.fit_on_texts(input_data['opcode'])
    scanning_progress.progress(80, text=f'{80}% | Converting texts to sequences...')
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

    scanning_progress.progress(85, text=f'{85}% | Getting BERT-BiLSTM Prediction...')
    y_pred_classes_m1, time_m1 = predict_model(BERT_BiLSTM, [input_data['source_code'], opcode_matrix], "BERT_BiLSTM")
    time.sleep(2)
    scanning_progress.progress(90, text=f'{90}% | Getting BERT-GNN Prediction...')
    y_pred_classes_m2, time_m2 = predict_model(BERT_GNN, [input_data['source_code'], gnn_input], "BERT_GNN")
    time.sleep(2)
    scanning_progress.progress(95, text=f'{95}% | Getting BiLSTM-GNN Prediction...')
    y_pred_classes_m3, time_m3 = predict_model(BiLSTM_GNN, [opcode_matrix, gnn_input], "BiLSTM_GNN")
    time.sleep(2)
    scanning_progress.progress(98, text=f'{98}% | Getting VulnSense Prediction...')
    y_pred_classes_vulnsense, time_vulnsense = predict_model(VulnSense, [input_data['source_code'], opcode_matrix, gnn_input], "VulnSense")

    scanning_progress.progress(99, text=f'{99}% | Getting Results...')
    time.sleep(freeze_time)
    scanning_progress.progress(100, text=f'{100}% | Showing Results...')
    scanning_progress.empty()
    
    return [y_pred_classes_m1[0], y_pred_classes_m2[0], y_pred_classes_m3[0], y_pred_classes_vulnsense[0]], [time_m1, time_m2, time_m3, time_vulnsense]

def set_page_configuration():   
    st.set_page_config(page_title='VulnSense - HomePage', layout='wide', page_icon='InsecLabLogo.png', initial_sidebar_state="collapsed")

def load_custom_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
@st.cache_resource
def load_model():
    VulnSense = tf.keras.models.load_model('Model/Model/VulnSense')
    BERT_BiLSTM = tf.keras.models.load_model('Model/Model/M1')
    BERT_GNN = tf.keras.models.load_model('Model/Model/M2')
    BiLSTM_GNN = tf.keras.models.load_model('Model/Model/M3')
    return VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN    
    

def create_header():
    header_col1, header_col2 = st.columns([0.07, 0.50])
    with header_col1:
        st.write("<h1 style='font-size: 36px; color: coral;'>VulnSense</h1>", unsafe_allow_html=True)
    with header_col2:
        pass

def display_homepage():
    st.subheader("| VulnSense - A Multimodal Deep Learning Approach for Efficient Vulnerability Detection in Smart Contracts")
    col1, col2 = st.columns(2, gap="small")
    with col1:
        st.image("Multimodal_Vulnsense.jpg")
        st.caption("VulnSense - Model Diagram", unsafe_allow_html=True)
    with col2:
        intro_text = """
        We present a comprehensive approach for efficient
        vulnerability detection in Ethereum smart contracts using a multimodal deep
        learning (DL) approach. Our proposed approach combines two levels of
        features in smart contracts, including source code, bytecode, and utilizes
        BERT and Bi-LSTM models to extract and analyze the features. The last layer
        of our multimodal approach is a fully connected layer that predicts the
        vulnerability in Ethereum smart contracts. We address the limitations of
        existing deep learning-based vulnerability detection methods for smart
        contracts, which often rely on a single type of feature or model, resulting
        in limited accuracy and effectiveness. The experimental results show that
        our proposed approach achieves superior results compared to existing
        state-of-the-art methods, demonstrating the effectiveness and potential of
        multimodal DL approaches in smart contract vulnerability detection.
        """
        st.write(f'<p style="color:#9c9d9f">{intro_text}</p>', unsafe_allow_html=True)

    col3, col4 = st.columns(2, gap="medium")
    with col3:
        st.subheader("+ Github")
        st.write(
            '<p style="color:#9c9d9f">Want to learn more? Visit <a href="https://github.com/bluesoju/VulnSense-UIT">GitHub</a> or our <a href="https://arxiv.org/abs/2309.08474">paper</a>.</p><br>',
            unsafe_allow_html=True,
        )
    with col4:
        st.subheader("+ Usage")
        st.write(
            '<p style="color:#9c9d9f">To begin the audit process, navigate to the <b>"Audit your contract now" tab.<b></p>',
            unsafe_allow_html=True,
        )

def is_valid_address(address):
    return bool(re.match(r'^0x[0-9a-fA-F]{40}$', address))

def display_contract_source_code():
    source_code = st.file_uploader("Upload a .sol file", type=["sol"])
    if source_code is not None:
        with st.expander("View Source Code"):
            raw_code = source_code.getvalue()
            formatted_code = textwrap.dedent(raw_code.decode())
            st.code(formatted_code, language="sol")
        return formatted_code
    return None

def security_score_gauge(progress):
    
    if progress > 60:
        text = "Your contract is at critical security risk "
        color = "red"
    elif progress > 20:
        text = "Your contract is dangerous but not at risk"
        color = "orange"
    else:
        text = "Your contract is safe"
        color = "green"
    
    fig = go.Figure()
    # Create the completed progress bar
    fig.add_trace(go.Indicator( 
        mode="gauge+number+delta",
        value=progress,  # Start with 0 for animation
        delta = {'reference': 20, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={'axis': {'range': [None, 100]}, 'bar': {'color': color}},
        number={'suffix': 'pts'},
        domain={'x': [0, 1], 'y': [0, 1]}
    ))

    fig.update_layout(height=200, margin=dict(l=20, r=20, t=20, b=20))

    progress_bar = st.plotly_chart(fig)

    progress_bar.write(fig)
    
    ## Animate the progress
    # for i in range(progress + 1):
    #     fig.data[0].value = i
    #     progress_bar.write(fig)
    #     time.sleep(0.03)  # Adjust the sleep time for the desired animation speed
    
    return text, color

def display_scan_results(results, secs, elapsed_time, graph=None, code=None):
    
    def render_cfg(graph):
        # Plot the graph
        graph_source = Source(''.join(graph), filename=f'cfg_graph', format='png')
        graph_source.render()  # This generates the PNG file
    
    render_cfg(graph[0])
    
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
        
    col1, col2 = st.columns(2, gap="small")
    with col1:
        progress_value, flag_count = calculate_score(results)
        text, color = security_score_gauge(progress_value)
        
        # Center-align the text within the container
        styled_text = f'<p style="color:{color}; text-align: center;">{text}</p>'
        
        # Display the text with the appropriate color and centered alignment
        st.markdown(styled_text, unsafe_allow_html=True)
    with col2:
        severity_counts = {
            'high': 0,
            'medium': 0,
            'critical': 0
        }
        for vul in results:
            if vul == 0:
                severity_counts['high'] += 1
            if vul == 2:
                severity_counts['medium'] += 1
                
        unique_non_one_values = [num for num, count in Counter(results).items() if num != 1 and count > 0]
        
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; text-align: center; margin-bottom: 30px;">
                <p style="color: {'red' if flag_count > 0 else 'green'}; background-color: {'rgba(255, 0, 0, 0.2)' if flag_count > 0 else 'rgba(0, 128, 0, 0.2)'}; padding: 5px; border-radius: 5px;">
                    {'⛔' if flag_count > 0 else '✅'} {flag_count} security vendor(s) flagged this contract as vulnerable.
                </p>
                <p style="margin-top: 10px;">There are {len(unique_non_one_values)} distinct types of identified vulnerabilities in the contract.</p>
                <table style="width:100%; margin-top: 10px;">
                    <tr>
                        <th>Severity</th>
                        <th>Count</th>
                    </tr>
                    <tr>
                        <td style="background-color: darkred; color: white;">Critical</td>
                        <td>{severity_counts['critical']}</td>
                    </tr>
                    <tr>
                        <td style="background-color: red; color: white;">High</td>
                        <td>{severity_counts['high']}</td>
                    </tr>
                    <tr>
                        <td style="background-color: orange;">Medium</td>
                        <td>{severity_counts['medium']}</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
    with st.expander("SCAN STATISTICS"):
        st.markdown(
            f"""
            <div style="border: 1px solid #ccc; border-radius: 10px; padding: 10px; margin-bottom: 50px; margin-top: 50px">
                <div style="padding: 10px; border: 1px solid rgba(0, 0, 0, 0.2); border-radius: 8px; margin-bottom: 5px;">
                    <p>Security Score<span style="font-weight: bold; float: right;">{progress_value:.2f}/100.00</span></p>
                    <hr style="border: none; border-top: 1px solid #ccc; margin: 10px 0;">
                    <p>Detection Time<span style="font-weight: bold; float: right;">{elapsed_time:.2f} seconds</span></p>
                    <hr style="border: none; border-top: 1px solid #ccc; margin: 10px 0;">
                    <p style="margin-bottom: 0;">Total lines inspected<span style="font-weight: bold; float: right;">{len(code.splitlines())}</span></p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    tab1, tab2 = st.tabs(["DETECTION", "DETAILS"])
    with tab1:
        
        prediction_text_format = {
            1: '✅ Undetected',
            2: '⚠️ Reentrancy',
            0: '⭕ Arithmetic'
        }
        
        col1, col2 = st.columns(2)
        with col1:
            data = {
                'Model Name': ["VulnSense", "BERT-BiLSTM", "BERT-GNN", "BiLSTM-GNN", "BERT", "BiLSTM", "GNN"],
                'Result': [prediction_text_format[results[3]], 
                           prediction_text_format[results[0]], 
                           prediction_text_format[results[1]], 
                           prediction_text_format[results[2]], 
                           "None", "None", "None"]
            }

            df = pd.DataFrame(data)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
        with col2:
            # Pseudo data for model prediction time in seconds
            model_names = ["VulnSense", "BERT-BiLSTM", "BERT-GNN", "BiLSTM-GNN", "BERT", "BiLSTM", "GNN"]
            prediction_times = [secs[3], secs[0], secs[1], secs[2], 0, 0, 0]

            chart_data = pd.DataFrame({'Model Name': model_names, 'Prediction Time (secs)': prediction_times})

            fig = px.bar(chart_data, x='Model Name', y='Prediction Time (secs)', color='Prediction Time (secs)',
                        labels={'Prediction Time (secs)': 'Prediction Time (secs)'},  color_continuous_scale=[[0, 'lightpink'], [1, 'red']])
            st.plotly_chart(fig, use_container_width=True)
            
    with tab2:
        def visualize_graph(data):
            G = nx.Graph()
            G.add_nodes_from(range(data.num_nodes))
            edge_index = data.edge_index.numpy()

            for i in range(edge_index.shape[1]):
                edge = (edge_index[0, i], edge_index[1, i])
                G.add_edge(*edge)

            # Create a Plotly figure
            fig = go.Figure()

            # Add nodes
            pos = nx.spring_layout(G)
            for node in G.nodes:
                x, y = pos[node]
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(size=10, color='skyblue'),
                    text=str(node),
                    showlegend=True
                ))

            # Add edges
            for edge in G.edges:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=1, color='lightcoral'),
                    showlegend=True
                ))

            # Customize layout
            fig.update_layout(
                hovermode='closest',
                showlegend=True,
                margin=dict(l=0, r=0, b=0, t=0)
            )

            # Display the Plotly figure using Streamlit
            st.plotly_chart(fig, use_container_width=True)    
        
        col1, col2 = st.columns(2, gap='medium')
        
        with col1:
            st.subheader("Contract CFG Visualization")
            # Load the image
            image_path = 'cfg_graph.png'
            original_image = Image.open(image_path)

            # Invert the colors
            inverted_image = ImageOps.invert(original_image)

            # Display the inverted image using streamlit
            st.image(np.array(inverted_image), use_column_width=True)
            
        with col2:
            st.subheader("Vulnerabilities Flow Visualization")
            visualize_graph(graph[1])


def get_contract_info(address):
    params = {
        'module': 'contract',
        'action': 'getsourcecode',
        'address': address,
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    return data['result']
            
def main():
    set_page_configuration()
    load_custom_css("style.css")
    VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN = load_model()
    create_header()
    tab1, tab2 = st.tabs(["Homepage", "Audit your contract !"])

    with tab1:
        display_homepage()

    with tab2:
        input_option = st.selectbox("Choose an Input Method:", ("Source Code File", "Smart Contract Address", "Test Cases"))

        if input_option == "Source Code File":
            source_code = display_contract_source_code()

            if source_code is not None:
                
                if st.button("Scan Vulnerabilities"):
                    start_time = time.time()  # Record the start time
                    scanning_progress = st.progress(0, text="Scanning Contract")
                    df, gnn, graph_data = preprocess(source_code, scanning_progress)
                    results, secs = predict(df, gnn, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN, scanning_progress)
                    elapsed_time = time.time() - start_time  # Calculate the elapsed time
                    display_scan_results(results, secs, elapsed_time, graph_data, source_code)

        elif input_option == "Smart Contract Address":
            address = st.text_input("Enter the Smart Contract Address")
            cleaned_address = address.strip()

            if cleaned_address:
                if is_valid_address(cleaned_address):
                    raw_info = get_contract_info(cleaned_address)
                    
                    with st.expander("View Source Code"):
                        raw_code = raw_info[0]['SourceCode']
                        st.code(raw_code, language="sol")
 
                    if st.button("Scan Vulnerabilities"):
                        start_time = time.time()  # Record the start time
                        scanning_progress = st.progress(0, text="Scanning Contract")
                        df, gnn, graph_data = preprocess(raw_code, scanning_progress)
                        results, secs = predict(df, gnn, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN, scanning_progress)
                        elapsed_time = time.time() - start_time  # Calculate the elapsed time
                        display_scan_results(results, secs, elapsed_time, graph_data, raw_code)
                else:
                    st.warning("Please enter a valid Smart Contract Address.")
            else:
                st.warning("Please enter a Smart Contract Address")
        elif input_option == "Test Cases":
            
            pre_gnn = None
            
            # Define the folder where the test cases are located
            test_cases_folder = "Tests"
            test_case = st.radio("Select a Test Case", ["clean_file", "reentrancy_file", "arithmetic_file"])
            if test_case:
                st.write(f"You selected: {test_case}")
                # Create the file path based on the selected test case  
                file_path = os.path.join(test_cases_folder, f"{test_case}.sol")
                # Check if the file exists
                if os.path.exists(file_path):
                    with st.expander(f"View Source Code"):
                        with open(file_path, "r") as test_case_file:
                            code_content = test_case_file.read()
                            st.code(code_content, language="sol")
                    
                    if st.button("Scan Vulnerabilities"):
                        start_time = time.time()  # Record the start time
                        scanning_progress = st.progress(0, text="Scanning Contract")
                        df, gnn, graph_data = preprocess(code_content, scanning_progress)
                        results, secs = predict(df, gnn, VulnSense, BERT_BiLSTM, BERT_GNN, BiLSTM_GNN, scanning_progress)
                            
                        elapsed_time = time.time() - start_time  # Calculate the elapsed time
                        display_scan_results(results, secs, elapsed_time, graph_data, code_content)
                else:
                    st.error("Test case file not found.")
                
                
            
if __name__ == "__main__":
    main()
