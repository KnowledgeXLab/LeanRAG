import csv
import json
from tools.logger_factory import setup_logger
import re


logger = setup_logger("triple")


class Triple:
    def __init__(self, head, relation, tail):
        self.head = head.strip().replace("<", "").replace(">", "")
        self.relation = relation.strip().replace("<", "").replace(">", "")
        self.tail = tail.strip().replace("<", "").replace(">", "")

    # Can use str(Triple) method to return triple, equivalent to __str__
    def __str__(self):
        return f"<{self.head}>\t<{self.relation}>\t<{self.tail}>"
    @classmethod
    def triple_json_format(self, triple, doc_name="", source_id=""):
        return {"triple": triple, "doc_name": doc_name, "source_id": source_id}

    @classmethod
    def get_example(self, entity, ref_kg_path):
        open_kg = []
        if entity:
            with open(ref_kg_path, "r", encoding="utf-8") as kgfile:
                for line in kgfile:
                    try:
                        triple = Triple(*line.strip().split("\t"))
                        if triple.head.lower() == entity.lower():
                            open_kg.append(str(triple))
                    except:
                        pass

        if open_kg != []:
            logger.info(f"Load open kg triple for {entity} nums: {len(open_kg)}.")
        else:
            open_kg = [
                "<Bacterial sulfate>\t<is a type of>\t<sulfur compound\n",
                "<Diabetes>\t<first line treatment>\t<Metformin>\n",
                "<Insulin>\t<drug type>\t<Long-acting analog>\n",
            ]
            # logger.info(f"use default triples.")
        return open_kg

    ## 对llm生成的三元组进行处理，并给出分析结果
    @classmethod
    def get_triple(self, entities, res, head_mode="acc"):
        """从大模型的回答中获得三元组，并分析异常情况"""

        # 保存LLM输出得到的所有三元组，形式为"xx | xx | xx"
        output_triples = set()
        error_triples = set()
        try:
            for item in res.split("\n"):
                item = re.sub(r"^\d+\.\s*", "", item, flags=re.MULTILINE)
                if len(item) > 0:
                    subs = item.split("|")
                    # error format tripple，格式错误， 不符合"xx | xx | xx"
                    if len(subs) < 3 or len(subs) > 3:
                        continue
                    # 获取head, ralation, tail, 创建三元组实例，并添加到output_triples
                    output_triples.add(Triple(*subs))
        except Exception as e:  # 捕获所有异常
            print("llm输出内容无法接解析成三元组：", e)  # 输出错误信息:

        output_triples = list(output_triples)

        for triple in output_triples:
            # acc (精确匹配模式): 这种模式要求三元组头部实体与节点完全一致（忽略大小写和首尾空格）
            if entities and head_mode == "acc":
                if triple.head.lower() not in [entity.lower() for entity in entities]:
                    error_triples.add(triple)
            elif triple.head.strip() == triple.tail.strip():
                error_triples.add(triple)

        error_triples = list(error_triples)

        ##移除不合规三元组
        for triple in error_triples:
            output_triples.remove(triple)

        # 获取新增三元组的尾实体
        head_entities = set([t.head for t in output_triples])
        tail_entities = set([t.tail for t in output_triples])

        return [str(item) for item in output_triples], head_entities, tail_entities

    @classmethod
    def parse_description_response(self, triple_str:str, response:str):
        """解析描述响应，生成六元组字符串"""
        # 处理制表符分隔的字符串并转换格式
        parts = triple_str.split('\t')
        cleaned_parts = [part.strip('<').strip('>').strip() for part in parts]  # 去除尖括号和空格

        try:
            data = json.loads(response.strip())
            
            # 提取描述信息
            subject_desc = data.get('subject', {}).get('description', '')
            relation_desc = data.get('relation', {}).get('description', '')
            object_desc = data.get('object', {}).get('description', '')
            
            head, relation, tail = cleaned_parts[0], cleaned_parts[1], cleaned_parts[2]
            
            # 组装六元组格式
            return f"<{head}>\t<{subject_desc}>\t<{relation}>\t<{relation_desc}>\t<{tail}>\t<{object_desc}>"
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Parse description response failed: {str(e)}, raw response: {response}")
            return triple_str
