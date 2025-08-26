from collections import Counter, defaultdict
from dataclasses import field
import json
import os
import time
import logging
import numpy as np
from openai import OpenAI
import pymysql
import tiktoken
from tqdm import tqdm
import yaml
from tools.utils import InstanceManager
from openai import OpenAI
from huggingface_hub import InferenceClient
from database_utils import build_vector_search, search_vector_search, find_tree_root, \
    search_nodes_link, search_nodes, search_community, search_chunks, get_text_units, find_path
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations

logger = logging.getLogger(__name__)
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['model']
EMBEDDING_URL = config['glm']['base_url']
EMBEDDING_PROVIDER = config['embedding']['provider']
HF_MODEL = config['huggingface']['model']
HF_TOKEN = config['huggingface']['HF_TOKEN']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def embedding(texts: list[str]) -> np.ndarray:
    """Embedding function that supports both GLM and HuggingFace providers."""
    if EMBEDDING_PROVIDER == "hf-inference":
        # Use HuggingFace cloud inference
        client = InferenceClient(
            provider=EMBEDDING_PROVIDER,
            api_key=HF_TOKEN,
        )

        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Use feature extraction to get embeddings
            result = client.feature_extraction(
                text=text,
                model=HF_MODEL
            )
            # The result can be a numpy array or list representing the embedding vector
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                # Convert numpy array to list if needed, then append
                if isinstance(result, np.ndarray):
                    embeddings.append(result.tolist())
                else:
                    embeddings.append(result)

        return np.array(embeddings)

    else:  # Default to GLM/OpenAI compatible provider
        model_name = EMBEDDING_MODEL
        client = OpenAI(
            api_key=EMBEDDING_MODEL,
            base_url=EMBEDDING_URL
        )
        embedding = client.embeddings.create(
            input=texts,
            model=model_name,
        )
        final_embedding = [d.embedding for d in embedding.data]
        return np.array(final_embedding)


tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


def get_reasoning_chain(global_config, entities_set):
    maybe_edges = list(combinations(entities_set, 2))
    reasoning_path = []
    reasoning_path_information = []
    db_name = global_config['working_dir'].split("/")[-1]
    information_record = []
    for edge in maybe_edges:
        a_path = []
        b_path = []
        node1 = edge[0]
        node2 = edge[1]
        node1_tree = find_tree_root(db_name, node1)
        node2_tree = find_tree_root(db_name, node2)

        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index, (i, j) in enumerate(zip(node1_tree, node2_tree)):
            if i == j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i != j:
                a_path.append(i)
                b_path.append(j)

        reasoning_path.append(a_path+[b_path[len(b_path)-1-i]
                              for i in range(len(b_path))])
        a_path = list(set(a_path))
        b_path = list(set(b_path))
        for maybe_edge in list(combinations(a_path+b_path, 2)):
            if maybe_edge[0] == maybe_edge[1]:
                continue
            information = search_nodes_link(
                maybe_edge[0], maybe_edge[1], global_config['working_dir'])
            if information == None:
                continue
            information_record.append(information)
            reasoning_path_information.append(
                [maybe_edge[0], maybe_edge[1], information[2]])
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information = list(
        set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description = "\n".join(
        temp_relations_information)
    return reasoning_path, reasoning_path_information_description


def get_entity_description(global_config, entities_set, mode=0):

    columns = ['entity_name', 'parent', 'description']
    entity_descriptions = "\t\t".join(columns)+"\n"
    entity_descriptions += "\n".join([information[0]+"\t\t"+information[1] +
                                     "\t\t"+information[2] for information in entities_set])

    return entity_descriptions


def get_aggregation_description(global_config, reasoning_path, if_findings=False):

    aggregation_results = []

    communities = set(
        [community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp = search_community(community, global_config['working_dir'])
        if temp == "":
            continue
        aggregation_results.append(temp)
    if if_findings:
        columns = ['entity_name', 'entity_description', 'findings']
        aggregation_descriptions = "\t\t".join(columns)+"\n"
        aggregation_descriptions += "\n".join([information[0]+"\t\t"+str(
            information[1])+"\t\t"+information[2] for information in aggregation_results])
    else:
        columns = ['entity_name', 'entity_description']
        aggregation_descriptions = "\t\t".join(columns)+"\n"
        aggregation_descriptions += "\n".join([information[0]+"\t\t"+str(
            information[1]) for information in aggregation_results])
    return aggregation_descriptions, communities


def query_graph(global_config, db, query):
    use_llm_func: callable = global_config["use_llm_func"]
    embedding: callable = global_config["embeddings_func"]
    b = time.time()
    level_mode = global_config['level_mode']
    topk = global_config['topk']
    chunks_file = global_config["chunks_file"]
    entity_results = search_vector_search(global_config['working_dir'], embedding([
                                          query])[0], topk=topk, level_mode=level_mode)
    v = time.time()
    res_entity = [i[0]for i in entity_results]
    chunks = [i[-1]for i in entity_results]
    entity_descriptions = get_entity_description(global_config, entity_results)
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(
        global_config, res_entity)
    # reasoning_path,reasoning_path_information_description=get_path_chain(global_config,res_entity)
    aggregation_descriptions, aggregation = get_aggregation_description(
        global_config, reasoning_path)
    # chunks=search_chunks(global_config['working_dir'],aggregation)
    text_units = get_text_units(
        global_config['working_dir'], chunks, chunks_file, k=5)
    describe = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
    e = time.time()

    # print(describe)
    sys_prompt = PROMPTS["rag_response"].format(context_data=describe)
    response = use_llm_func(query, system_prompt=sys_prompt)
    g = time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")

    print(f"response time: {g-e:.2f}s")
    return describe, response


if __name__ == "__main__":
    db = pymysql.connect(host='localhost', user='root', port=4321,
                         passwd='123', database='leanrag_default', charset='utf8mb4')
    global_config = {}
    WORKING_DIR = "datasets/mix/mix_chunk"
    global_config['chunks_file'] = "datasets/mix/mix_chunk/mix_chunk.json"
    global_config['embeddings_func'] = embedding
    global_config['working_dir'] = WORKING_DIR
    global_config['topk'] = 10
    global_config['level_mode'] = 1
    # Use OpenRouter configuration similar to build_graph.py
    instanceManager = InstanceManager(
        url="https://openrouter.ai/api",
        ports="api",
        gpus=[0],
        generate_model="openai/gpt-5-mini",
        startup_delay=30
    )

    global_config['use_llm_func'] = instanceManager.generate_text
    query = "what happened to Sir Henry de Bohun at the Battle of Bannockburn?"
    topk = 10
    ref, response = query_graph(global_config, db, query)
    print(ref)
    print("#"*20)
    print(response)
    db.close()
