import re
from solcx import compile_standard, install_solc
from binary_extractor.platforms.ETH.cfg import EthereumCFG
from binary_extractor.analysis.graph import CFGGraph
import numpy as np

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd

def getVersionPragma(source_code):                     # param la path cua contract          
    data = source_code.split('\n')
    for line in data:                               # duyet tung dong trong contract do
        if 'pragma' in line and 'solidity' in line:                        # neu dong do chua chu "pragma"
            temp = line.split()                     # chuyen dong do thanh list ['pragma', 'solidity', '^0.4.19;']
            if len(temp) == 3 and temp[2][0].isnumeric() == True:       # ['pragma', 'solidity', '0.4.19;']
                return temp[2][0:-1]
            elif len(temp) == 3 and temp[2][1].isnumeric() == True:       # ['pragma', 'solidity', '^0.4.19;']
                return temp[2][1:-1]
            elif len(temp) == 3 and temp[2][1].isnumeric() == False:    # ['pragma', 'solidity', '<=0.4.19;']
                return temp[2][2:-1]
            elif len(temp) > 3 and len(temp[2]) == 1:
                return temp[2][-1]
            elif len(temp) == 4 and temp[2][1].isnumeric() == True:      # ['pragma', 'solidity', '>0.4.22', '<0.6.0]
                return temp[2][1:]
            elif len(temp) == 4 and temp[2][1].isnumeric() == False:     # ['pragma', 'solidity', '>=0.4.22', '<0.6.0]
                return '0.5.0'
    return '0.4.22'

def clean_opcode(opcode_str):
    # remove hex characters (0x..)
    opcode_str = re.sub(r'0x[a-fA-F0-9]+', '', opcode_str)
    
    # remove newline characters
    opcode_str = opcode_str.replace('\n', ' ')
    
    # extract only the opcode names
    opcodes = re.findall(r'[A-Z]+', opcode_str)
    
    return opcodes

def remove_comment(source_code):
    print(f'Removing comments...')
    source_code = re.sub(r"//\*.*", "", source_code)
    source_code = re.sub(r"#.*", "", source_code)
    # Remove multi-line comments
    source_code = re.sub(r"/\*.*?\*/", "", source_code, flags=re.DOTALL)
    source_code = re.sub(r"\"\"\".*?\"\"\"/", "", source_code, flags=re.DOTALL)
    
    source_code = re.sub(r"//.*", "", source_code)

    print(f'Removing redundant spaces and tabs...')
    # Remove redundant spaces and tabs
    source_code = re.sub(r"[\t ]+", " ", source_code)

    print(f'Removing empty lines...')
    # Remove empty lines
    source_code = re.sub(r"^\s*\n", "", source_code, flags=re.MULTILINE)
    return source_code

def get_bytecode(regex, bytecode):
    cc = bytecode.split(regex)
    bytecode = ''.join(cc)
    match = re.findall(r"__.{1,50}_", bytecode)
    if len(match) != 0:
        bytecode = get_bytecode(match[0], bytecode)
        return bytecode
    else:
        return bytecode

def return_bytecode_opcode(content):
    version = getVersionPragma(content)
    print(f'Getting pragma version: {version}')
    try:
        install_solc(version)
    except:
        version = '0.4.11'
        install_solc(version)
    try:
        compiled_sol = compile_standard(
            {
                "language": "Solidity",
                "sources": {'cc': {"content": content}},
                "settings": {
                    "outputSelection": {
                        "*": {
                            "*": ["evm.bytecode.opcodes", "metadata", "evm.bytecode.sourceMap"]
                        }
                    }
                },
            },
            solc_version=version,
        )
    except:
        compiled_sol = compile_standard(
            {
                "language": "Solidity",
                "sources": {'cc': {"content": content}},
                "settings": {
                    "outputSelection": {
                        "*": {
                            "*": ["evm.bytecode.opcodes", "metadata", "evm.bytecode.sourceMap"]
                        }
                    }
                },
            },
            solc_version='0.4.24',
        )

    contracts_name = compiled_sol["contracts"]['cc'].keys()
    list_opcode = []
    list_bytecode = []
    for contract in contracts_name:
        try:
            bytecode = compiled_sol["contracts"]["cc"][contract]["evm"]["bytecode"]["object"]
            opcode = compiled_sol["contracts"]["cc"][contract]["evm"]["bytecode"]["opcodes"]

            match = re.findall(r"__.{1,50}_", bytecode)
            if len(match) != 0:
                bytecode = get_bytecode(match[0], bytecode)

            list_bytecode.append(bytecode)
            list_opcode.append(opcode)
        except KeyError:
            # Handle the case where bytecode or opcode is not available
            print(f"Skipping contract {contract} as bytecode or opcode is not available.")
            continue
        except Exception as e:
            # Handle any other exceptions that may occur
            print(f"An error occurred while processing contract {contract}: {str(e)}")
            continue
    
    final_bytecode = ''.join(list_bytecode)
    final_opcode = ''.join(list_opcode)
    final_opcode = clean_opcode(final_opcode)
    final_opcode = ''.join(list_opcode)

    return final_bytecode, final_opcode

def bytecode_to_cfg(bytecode_hex):
    # create the CFG
    cfg = EthereumCFG(bytecode_hex)
    graph = CFGGraph(cfg)
    graph.view()
    
    with open('graph.cfg.gv', 'r') as f:
        lines = f.readlines()
    
    nodes = []
    edges = []
    
    flag = 0
    for line in lines:
        if "block" in line and flag == 0:
            nodes.append(re.sub(r'\\l', r' ', line).strip())
        if "block" in line and flag != 0:
            edges.append(line.strip())
        if "}" in line:
            flag += 1
            continue
        
    return lines, nodes, edges