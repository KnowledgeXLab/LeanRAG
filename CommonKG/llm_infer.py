from collections import deque
from tqdm import tqdm
import requests
import json
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
import time
from openai import OpenAI

from tools.logger_factory import setup_logger
from CommonKG.triple import Triple

logger = setup_logger("llm_processor")

class InstanceManager:
    def __init__(self, ports, gpus, base_url, startup_delay=5):
        self.ports = ports
        self.gpus = gpus
        self.base_url = base_url
        self.instances = []
        self.lock = threading.Lock()
        self.current_instance = 0  # Used for polling strategy

        for port, gpu in zip(self.ports, self.gpus):
            self.start_instance(gpu, port)
            self.instances.append({"port": port, "load": 0})
        time.sleep(startup_delay)  # Wait for all instances to start

    def start_instance(self, num, port):
        """Start ollama instance on specific GPU and port"""
        # cmd = f"CUDA_VISIBLE_DEVICES={num} OLLAMA_HOST={self.base_url}:{port} ollama serve"
        cmd = f"OLLAMA_HOST={self.base_url}:{port} ollama serve"
        print("Running command:", cmd)
        # subprocess.Popen(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    def get_available_instance(self):
        """Use polling strategy to get an available instance"""
        with self.lock:
            instance = self.instances[self.current_instance]
            self.current_instance = (self.current_instance + 1) % len(self.instances)
            return instance["port"]  # Return port

    def generate_text(self, prompt, model="qwen:72b", temperature=0, output_json=False):
        """Send request to selected instance"""
        port = self.get_available_instance()
        base_url = f"{self.base_url}:{port}"
        
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "temperature": temperature},
            timeout=30  # Set timeout to avoid infinite waiting
        )
        response.raise_for_status()
        return response



class LLM_Processor:
    def __init__(self, args):
        self.model = args["llm_model"]
        self.base_url = args["llm_url"]
        self.api_key = args["llm_api_key"]
        self.max_error = args["max_error"]
        self.ports = [8001 + i for i in range(args["gpu_nums"])]  # Port pool
        self.gpus = [i for i in range(args["gpu_nums"])]  # GPU numbers
        if args["use_ollama"]:
            print(f'OLLAMA was requested: {args["use_ollama"]}')
            self.manager = InstanceManager(self.ports, self.gpus, self.base_url)
            self.generate_text = self.manager.generate_text
        elif args["use_vllm"]:
            print(f'VLLM was requested: {args["use_vllm"]}')
            self.manager = InstanceManager(self.ports, self.gpus, self.base_url)
            self.generate_text = self.vllm_generate_text
        else:
            print(f'All else has failed, final default being executed for either LLM/VLLM by creating general default text: {self.default_generate_text}')
            print(f"Using model: {self.model} at {self.base_url} with api key: {self.api_key[:4]}****")
            self.generate_text = self.default_generate_text
    

    def vllm_generate_text(self, prompt, model, temperature=0, max_tokens=4096, output_json=False):
        """Generate text using vLLM"""
        
        port = self.manager.get_available_instance()
        base_url = f"{self.base_url}:{port}/v1"
        
        try:
            if output_json:
                # Call Chat Completion API and set parameters
                 response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens" : max_tokens,
                "response_format":{"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False}
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120
        )
               
            else:
                 response = requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens" : max_tokens,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120
        )
            response.raise_for_status()
            res=json.loads(response.content)
            response_message = res["choices"][0]["message"]['content']

            return response_message
            
        except Exception as e:
            logger.info(f"Error: {e}")
            return None

    def default_generate_text(self, prompt, model="qwen/qwen3-32b", temperature=0, max_tokens=4096, output_json=False):
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        print(f"default_generate_text API Key: {self.api_key}")
        print(f"Using model: {model} at {self.base_url}")
        print(f"Prompt: {prompt}...type: {type(prompt)}")  # Print first 50 characters of the prompt for debugging
        try:
            if output_json:
                # Call Chat Completion API and set parameters
                completion = client.chat.completions.create(
                    model = model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}],
                    max_tokens = max_tokens,
                    response_format={"type": "json_object"}
                    )
            else:
                completion = client.chat.completions.create(
                    model = model,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}],
                    max_tokens = max_tokens
                    )
                
            print(f"LLM response: {completion} type: {type(completion)}")    
            response = completion.model_dump()["choices"][0]["message"]["content"]
            print(f'completion response: {response}')
            return  response
            
        except Exception as e:
            logger.info(f"Error: {e}")
            return None
    
    def extract_responses(self, response):
        """Extract text from streaming response."""
        responses = []
        for line in response.iter_lines():
            if line:
                line_data = json.loads(line.decode("utf-8"))
                responses.append(line_data["response"])
        return "".join(responses)
        

    def extract_triple_prompt(self, corpus: str, entities: list, ref_kg_path):

        if len(entities) > 0:
            for entity in entities:
                example_triple_list = Triple.get_example(entity, ref_kg_path)
                example_triple = (
                    "\n".join(
                        [f"{i+1}. {item.replace(chr(9), ' | ')}" for i, item in enumerate(example_triple_list)]
                    )
                    + "\n"
                )

            entities_str = ",".join(entities)

            prompt = (
                f'\n[Text]:\n "{corpus}"\n\n'
                f"[Instruction]:\n A triple is composed of a subject, a predicate, and an object. " 
                f"Please extract all triples related to [{entities_str}] from the above text as much as possible. \n"
                f"The triples must have one of  [{entities_str}] as the head entity, the text of the tail entity must be short, "
                f"and be output strictly in triple format.\n\n"
                f"[Examples]:\n"
                f"{example_triple}\n"
            )
        else:
            example_triple_list = Triple.get_example("", ref_kg_path)
            example_triple = (
                    "\n".join(
                        [f"{i+1}. {item.replace(chr(9), ' | ')}" for i, item in enumerate(example_triple_list)]
                    )
                    + "\n"
                )

            prompt = (
                f'\n[Text]:\n "{corpus}"\n\n'
                f"[Instruction]:\n A triple is composed of a subject, a predicate, and an object. " 
                f"The subject and object are entities and must be short."
                f"Please extract all triples from the above text as much as possible. \n"
                f"Entities include proper nouns, discipline terminologies, abstract and collective nouns, etc. "
                f"Entities Do Not include any verbs orwords without specific meanings such as time, location, number, measurement, etc. \n"
                f"and be output strictly in triple format.\n\n"
                f"[Examples]:\n"
                f"{example_triple}\n"
            )
        return prompt
    

    def extract_description_prompt(self, text:str, triple:str):

        # Process tab-separated string and convert format
        parts = triple.split('\t')
        cleaned_parts = [part.strip('<').strip('>').strip() for part in parts]  # Remove angle brackets and spaces
        triple_str = f"subject: {cleaned_parts[0]}, relation: {cleaned_parts[1]}, object: {cleaned_parts[2]}"

    
        prompt = f'''
        [Text]:
        {text}

        [Triple]:
        {triple_str}

        [Instruction]:
        Each triple consists of a subject, predicate, and object. Based on the triple and the above text fragment (the triple extracted source), extract:

        - The subject and object entities with the following fields:
            - "name"
            - "description": concise summary (≤ 50 English words) capturing key attributes mentioned in the text

        - The relation with:
            - "name"
            - "description": explain the semantic meaning of the relation in context (≤ 50 English words)

        Special requirements:
        1. Prioritize using exact phrases from the text for name fields
        2. Relation description should explain **why** the connection exists based on text evidence

        [Output format]:
        {{
        "subject": {{
            "name": "xxx",
            "description": "xxx"
        }},
        "relation": {{
            "name": "xxx",
            "description": "xxx"
        }},
        "object": {{
            "name": "xxx",
            "description": "xxx"
        }}
        }}
        '''

        return prompt


    def call_api(self, user_prompt: str, system_prompt: str = "", output_json=False) -> str:
        response = self.generate_text(user_prompt, model=self.model, temperature=0, output_json=output_json)

        if not isinstance(response, str):  ## If not str, extract content from response
            response = self.extract_responses(response).strip()

        return response.replace("_", " ")


    def infer(self, prompt, output_json=False):
        error_count = deque(maxlen=self.max_error)
        cnt = 0
        while sum(error_count) < self.max_error:
            cnt += 1
            try:
                response= self.call_api(user_prompt=prompt, output_json=output_json)
                if not isinstance(response, str):
                    if isinstance(response, dict) and "data" in response:
                        response = response["data"]["output"]
                return response
            except (Exception, KeyboardInterrupt) as e:
                error_count.append(1)  # Record one error
                logger.info(f"LLM request error:{cnt}, {e}")

        if sum(error_count) >= self.max_error:
            logger.info(f"Maximum error tolerance reached, skipping text id: {id}")

    def entity_evaluate(self, entities):
        entities = "\n".join(entities)
        user_prompt = (
            "Analyze the entity list I provide and extract all entities (e.g., proper nouns, discipline terminologies, abstract and collective nouns, etc.). Follow these rules STRICTLY:\n"
            "1. Extract as many nouns from the text as possible\n"
            "2. Do not include any verbs or words without specific meanings such as time, location, number, measurement, etc.\n"
            "3. Output EXACTLY one entity per line\n"
            "4. Maintain ORIGINAL spelling/case from the input\n"
            "5. Ensure that the entities are not repeated\n"

            "Input entity list:\n"
            f"{entities}\n"

            "Now process this list. \n"
        )

        response = self.call_api(user_prompt)
        verify_entities = response.split("\n")
        verify_entities = list(set([item.strip() for item in verify_entities if len(item.strip()) > 0]))
        return verify_entities
