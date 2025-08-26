import json
import os
import sys
sys.path.append("/workspaces/leangraph")
from tqdm import tqdm
from prompt import PROMPTS
import tiktoken
import sys 
from tools.utils import read_jsonl, write_jsonl, create_if_not_exist,InstanceManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
threshold=50
def summarize_entity(entity_name, description, summary_prompt, threshold, tokenizer):
    tokens = len(tokenizer.encode(description))
    if tokens > threshold:
        exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
        response = use_llm(exact_prompt)
        return entity_name, response
    return entity_name, description  # 不需要摘要则返回原始 description



def process_triple(file_path,output_path):
    create_if_not_exist(output_path)
    triple_path=os.path.join(file_path,f"new_triples_{os.path.basename(file_path)}_descriptions.jsonl")
    
    with open(triple_path,"r") as f:
        entities={}
        relations=[]
        for uline in f:
            line=json.loads(uline)
            # print(f"Processing line: {line}")  # Debugging line to check content
            triple=line['triple'].split("\t")
            if len(triple) < 6:
                # print(f"Skipping malformed triple: {line['triple']}")
                continue
            doc_name=line['doc_name']
            source_id=line['source_id']
            head_entity=triple[0][1:-1]
            head_description=triple[1][1:-1]
            relation=triple[2][1:-1]
            relation_description=triple[3][1:-1]
            tail_entity=triple[4][1:-1]
            tail_description=triple[5][1:-1]
            
            if head_entity not in entities.keys():
                entities[head_entity]=dict(
                    entity_name=str(head_entity),
                    description=head_description,
                    source_id=source_id,
                    doc_name=doc_name,
                    degree=0,
                )
            else:
                entities[head_entity]['description']+=" | "+ head_description
                if entities[head_entity]['source_id']!= source_id:
                    entities[head_entity]['source_id']+= "|"+source_id
            if tail_entity not in entities.keys():
                entities[tail_entity]=dict(
                    entity_name=str(tail_entity),
                    description=tail_description,
                    source_id=source_id,
                    doc_name=doc_name,
                    degree=0,
                )
            else:
                entities[tail_entity]['description']+=" | "+ tail_description
                if entities[tail_entity]['source_id']!= source_id:
                    entities[tail_entity]['source_id']+= "|"+source_id
            relations.append(dict(
                src_tgt=head_entity,
                tgt_src=tail_entity,
                source=relation,
                description=relation_description,
                weight=1,
                source_id=source_id
            ))
    write_jsonl(relations,f"{output_path}/relation.jsonl") 
    res_entity=[]           
    tokenizer = tiktoken.get_encoding("cl100k_base")
    to_summarize = []
    summary_prompt=PROMPTS['summary_entities']
    for k,v in entities.items():
        v['source_id']="|".join(set(v['source_id'].split("|")))
        description=v['description']
        tokens = len(tokenizer.encode(description))
        if tokens > threshold:
            to_summarize.append((k, description))
        else:
            res_entity.append(v)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(summarize_entity, k, desc, summary_prompt, threshold, tokenizer): k
            for k, desc in to_summarize
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing descriptions"):
            k, summarized_desc = future.result()
            entities[k]['description'] = summarized_desc
            res_entity.append(entities[k])

    # Debugging: Check if res_entity has content before writing
    if not res_entity:
        print("Warning: No entities were processed. The 'res_entity' list is empty, so 'entity.jsonl' will not be created.")
    else:
        print(f"Processed {len(res_entity)} entities. Writing to entity.jsonl...")
        write_jsonl(res_entity,f"{output_path}/entity.jsonl")
        print(f"Successfully wrote to {output_path}/entity.jsonl")
        
if __name__=="__main__":
    MODEL = "qwen3_32b"
    num=4
    instanceManager=InstanceManager(
        url="https://openrouter.ai/api",
        ports="api",
        gpus=[0],
        generate_model="openai/gpt-5-mini",
        startup_delay=30
    )
    use_llm=instanceManager.generate_text
    file_path="/workspaces/leangraph/datasets/mix/mix_chunk"
    output_path="/workspaces/leangraph/datasets/mix/mix_chunk"
    process_triple(file_path,output_path)