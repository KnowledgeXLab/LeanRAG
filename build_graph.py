import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import field
import json
import os
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import yaml
from openai import AsyncOpenAI, OpenAI
from huggingface_hub import InferenceClient
from _cluster_utils import Hierarchical_Clustering
from tools.utils import write_jsonl, InstanceManager
from database_utils import build_vector_search, create_db_table_mysql, insert_data_to_mysql
import requests
import multiprocessing
logger = logging.getLogger(__name__)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL = config['deepseek']['model']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['model']
EMBEDDING_URL = config['glm']['base_url']
EMBEDDING_PROVIDER = config['embedding']['provider']
HF_MODEL = config['huggingface']['model']
HF_TOKEN = config['huggingface']['HF_TOKEN']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def get_common_rag_res(WORKING_DIR):
    entity_path = f"{WORKING_DIR}/entity.jsonl"
    relation_path = f"{WORKING_DIR}/relation.jsonl"
    # i=0
    e_dic = {}
    with open(entity_path, "r")as f:
        for xline in f:

            line = json.loads(xline)
            entity_name = str(line['entity_name'])
            description = line['description']
            source_id = line['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name] = dict(
                    entity_name=str(entity_name),
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description'] += "|Here is another description : " + description
                if e_dic[entity_name]['source_id'] != source_id:
                    e_dic[entity_name]['source_id'] += "|"+source_id

    #         i+=1
    #         if i==1000:
    #             break
    # i=0
    r_dic = {}
    with open(relation_path, "r")as f:
        for xline in f:

            line = json.loads(xline)
            src_tgt = str(line['src_tgt'])
            tgt_src = str(line['tgt_src'])
            description = line['description']
            weight = 1
            source_id = line['source_id']
            r_dic[(src_tgt, tgt_src)] = {
                'src_tgt': str(src_tgt),
                'tgt_src': str(tgt_src),
                'description': description,
                'weight': weight,
                'source_id': source_id
            }
            # e_dic[src_tgt]['degree']+=1
            # e_dic[tgt_src]['degree']+=1
            # i+=1
            # if i==1000:
            #     break

    return e_dic, r_dic


def embedding(texts: list[str]) -> np.ndarray:
    """Embedding function that supports both GLM and HuggingFace providers."""
    if EMBEDDING_PROVIDER == "hf-inference":
        # Use HuggingFace cloud inference
        client = InferenceClient(
            provider="hf-inference",
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
            # The result can be a numpy array or list of floats representing the embedding vector
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                # Convert numpy array to list for consistency
                embedding_vec = result.tolist() if isinstance(result, np.ndarray) else result
                embeddings.append(embedding_vec)

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


def embedding_init(entities: list[dict]) -> list[dict]:
    """Initialize embeddings for entities using the configured embedding provider."""
    texts = [truncate_text(i['description']) for i in entities]

    if EMBEDDING_PROVIDER == "hf-inference":
        # Use HuggingFace cloud inference
        client = InferenceClient(
            provider="hf-inference",
            api_key=HF_TOKEN,
        )

        embeddings = []
        # First, try to get one embedding to determine the dimension
        embedding_dim = 1024  # Default for BGE-M3

        for i, text in enumerate(texts):
            try:
                print(f"\n=== Embedding Entity {i} ===")
                print(
                    f"Input text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"Using model: {HF_MODEL}")

                # Use feature extraction to get embeddings
                result = client.feature_extraction(
                    text=text,
                    model=HF_MODEL
                )

                print(f"Raw API result type: {type(result)}")
                print(
                    f"Raw API result: {str(result)[:200]}{'...' if len(str(result)) > 200 else ''}")

                # The result can be a numpy array or list of floats representing the embedding vector
                if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                    print(
                        f"✓ Success: Got embedding with dimension {len(result)}")
                    # Convert numpy array to list for consistency
                    embedding_vec = result.tolist() if isinstance(result, np.ndarray) else result
                    embeddings.append(embedding_vec)
                    if i == 0:  # Update embedding dimension based on first successful result
                        embedding_dim = len(embedding_vec)
                else:
                    # Create a default zero embedding if the result is invalid
                    print(
                        f"⚠ Warning: Empty embedding result for entity {i}, using zero vector")
                    print(f"Result was: {result}")
                    embeddings.append([0.0] * embedding_dim)
            except Exception as e:
                print(f"✗ Error getting embedding for entity {i}: {e}")
                print(f"Exception type: {type(e)}")
                # Create a default zero embedding on error
                embeddings.append([0.0] * embedding_dim)

        final_embedding = embeddings

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

    # Ensure we have the same number of embeddings as entities
    if len(final_embedding) != len(entities):
        print(
            f"Warning: Mismatch between entities ({len(entities)}) and embeddings ({len(final_embedding)})")
        # Pad with zero vectors if needed
        embedding_dim = len(final_embedding[0]) if final_embedding else 1024
        while len(final_embedding) < len(entities):
            final_embedding.append([0.0] * embedding_dim)

    for i, entity in enumerate(entities):
        entity['vector'] = np.array(final_embedding[i])
    return entities


tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_text(text, max_tokens=4096):
    """Truncate text to a maximum number of tokens."""
    if not text or text.strip() == "":
        return "No description available"  # Provide default text for empty descriptions

    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


def embedding_data(entity_results):
    entities = [v for k, v in entity_results.items()]
    entity_with_embeddings = []
    embeddings_batch_size = 64
    num_embeddings_batches = (
        len(entities) + embeddings_batch_size - 1) // embeddings_batch_size

    batches = [
        entities[i * embeddings_batch_size: min(
            (i + 1) * embeddings_batch_size, len(entities))]
        for i in range(num_embeddings_batches)
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embedding_init, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            entity_with_embeddings.extend(result)

    for i in entity_with_embeddings:
        entiy_name = i['entity_name']
        vector = i['vector']
        entity_results[entiy_name]['vector'] = vector
    return entity_results


def hierarchical_clustering(global_config):
    entity_results, relation_results = get_common_rag_res(
        global_config['working_dir'])
    all_entities = embedding_data(entity_results)
    hierarchical_cluster = Hierarchical_Clustering()
    all_entities, generate_relations, community = hierarchical_cluster.perform_clustering(global_config=global_config, entities=all_entities, relations=relation_results,
                                                                                          WORKING_DIR=WORKING_DIR, max_workers=global_config['max_workers'])
    try:
        all_entities[-1]['vector'] = embedding(all_entities[-1]['description'])
        build_vector_search(all_entities, f"{WORKING_DIR}")
    except Exception as e:
        print(f"Error in build_vector_search: {e}")
    for layer in all_entities:
        if type(layer) != list:
            if "vector" in layer.keys():
                del layer["vector"]
            continue
        for item in layer:
            if "vector" in item.keys():
                del item["vector"]
            if len(layer) == 1:
                item['parent'] = 'root'
    save_relation = [
        v for k, v in generate_relations.items()
    ]
    save_community = [
        v for k, v in community.items()
    ]
    write_jsonl(save_relation,
                f"{global_config['working_dir']}/generate_relations.json")
    write_jsonl(save_community,
                f"{global_config['working_dir']}/community.json")
    create_db_table_mysql(global_config['working_dir'])
    insert_data_to_mysql(global_config['working_dir'])


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str,
                        default="datasets/mix/mix_chunk")
    parser.add_argument("-n", "--num", type=int, default=2)
    args = parser.parse_args()

    WORKING_DIR = args.path
    num = args.num

    # For OpenRouter, we need to use a different configuration
    instanceManager = InstanceManager(
        # Remove the trailing slash and use the correct base
        url="https://openrouter.ai/api",
        ports=[""],  # Empty port since OpenRouter doesn't use ports
        gpus=[0],  # Single instance for API
        generate_model="openai/gpt-4o-mini",  # Fix the model name
        startup_delay=30
    )
    global_config = {}
    global_config['max_workers'] = num*4
    global_config['working_dir'] = WORKING_DIR
    global_config['use_llm_func'] = instanceManager.generate_text
    global_config['embeddings_func'] = embedding
    global_config["special_community_report_llm_kwargs"] = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)
