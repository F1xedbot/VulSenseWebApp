import jraph
import time
from openai.embeddings_utils import cosine_similarity, get_embedding as _get_embedding
from tenacity import  stop_after_attempt, wait_random_exponential
import openai
from typing import List, Tuple
import numpy as np

openai.api_key = "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"

get_embedding = _get_embedding.retry_with(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))


FEATURE_NUM = 1536
EDGE_FEATURE_NUM = 4

def get_adj_dict(edges: str) -> List[dict]:
    adj_dict = {}
    edge_id = 0
    edge_id_dict = {}
    edge_type = {"red": 0, "green": 1, "blue": 2, "cyan": 3}
    for line in edges:
        if "block" in line:
            # Split the line into block_id and block_dest_id
            block_id, block_dest_id = line.split(" -> ")

            # Cleaning to get block_id, block_dest_id, edge_label
            block_id = block_id.strip()
            block_dest_id, edge_label = block_dest_id.split("[color=")
            block_dest_id = block_dest_id.strip()
            edge_label = edge_label.strip()[:-1]


            if block_id not in adj_dict:
                adj_dict[block_id] = []
            adj_dict[block_id].append((block_dest_id, edge_type[edge_label]))
    return adj_dict

def get_x_dict(nodes: str) -> List[dict]:
    X_dict = {}
    for line in nodes:
        if "block" in line:
            block_id, label = line.split(" [label=")
            block_id = block_id.strip()
            label = label.strip().strip('"')
            label = label[:-2].strip()
            X_dict[block_id] = label
    return X_dict

def encoder(code: str) -> np.ndarray:
    text = "This is a block of EVM bytecode: " + code
    model_id = "text-embedding-ada-002"
    emb = openai.Embedding.create(input=[text], model=model_id)['data'][0]['embedding']
    emb_np = np.array(emb)
    return emb_np

def make_jraph_dataset(nodes, edges):
    x_dict = get_x_dict(nodes)
    adj_dict = get_adj_dict(edges)
    COUNT = 0
    for key in adj_dict.keys():
            if key not in x_dict.keys():
                raise Exception(f"Key {key} not in x_dict")
        
    blk_id_to_index = {blk_id: i for i, (blk_id, _) in enumerate(x_dict.items())}
    nodes = []
    edges = []
    senders = []
    receivers = []
    dataset = []

    # Convert dicts to jraph representation
    for i, (blk,code) in enumerate(x_dict.items()):
        COUNT = COUNT + 1
        t = time.localtime()
        current_time = time.strftime("%S", t)
        if current_time == 0:
            COUNT = 0
        if COUNT >= 458 and int(current_time) < 59:
            time.sleep(65 - int(current_time))
            COUNT = 0
            print(f"Waiting for {65 - int(current_time)}")
        nodes.append(encoder(code))
        if blk in adj_dict:
            for (dest, edge_type) in adj_dict[blk]:
                edge_one_hot = np.zeros((EDGE_FEATURE_NUM,))
                edge_one_hot[edge_type] = 1
                edges.append(edge_one_hot)
                senders.append(blk_id_to_index[blk])
                receivers.append(blk_id_to_index[dest])
    
    # Convert to jraph
    graph = jraph.GraphsTuple(
                                n_node=np.array([len(nodes)]),
                                n_edge=np.array([len(edges)]),
                                nodes=np.array(nodes), 
                                edges=np.array(edges), 
                                senders=np.array(senders), 
                                receivers=np.array(receivers), 
                                globals=np.array([1]),
                            )
    target = [1]
    dataset.append({"input_graph": graph, "target": target})
    return dataset