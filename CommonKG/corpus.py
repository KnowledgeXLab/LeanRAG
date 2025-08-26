from ahocorasick import Automaton
from tools.logger_factory import setup_logger
from tools import utils


logger = setup_logger("corpus_batch")


# class Corpus(object):
#     def __init__(self, doc_name, page_id, paragraph_id, corpus):
#         self.doc_name = doc_name
#         self.page_id = page_id
#         self.paragraph_id = paragraph_id
#         self.corpus = corpus

#     def get_match_words(self, entities: list):
#         match_words = {"doc_name": self.doc_name, "page_id": self.page_id, "paragraph_id": self.paragraph_id, "text": self.corpus, "match_words": []}
#         blacklist = ["table. ", "tab. ", "fig. ", "figure. "]
#         match_words["match_words"] = self.auto_match(entities)

#         return match_words

#     def auto_match(self, entities, lower_case=True):
#         entities = list(set(entities)) ## Remove duplicates
#         match_words = set()
#         A = Automaton()
#         for entity in entities:
#             # Chinese-English compatible lowercase conversion, replace keyword.lower() with custom function custom_lower_fast(keyword)
#             entity_key = utils.custom_lower_fast(entity) if lower_case else entity
#             # Retrieve entity, output corresponding (subject, entity)
#             A.add_word(entity_key, entity)
#         A.make_automaton()  # Build automaton
#         # Initialize match_raw: idx records text id, text_len records text length, unique_count records number of matched entities
#         _text = utils.custom_lower_fast(self.corpus) if lower_case else self.corpus
#         try:
#             for end_index, entity in A.iter(_text):
#                 end_index += 1
#                 start_index = end_index - len(entity)
#                 # If detected boundary is not a word boundary, skip
#                 if utils.is_word_boundary(_text, start_index, end_index):
#                     match_words.add(entity)
#         except Exception as e:
#             pass
#         return list(match_words)
class Corpus(object):
    def __init__(self, doc_name, source_id,  corpus):
        self.doc_name = doc_name
        self.source_id = source_id
        self.corpus = corpus

    def get_match_words(self, entities: list):
        match_words = {"doc_name": self.doc_name, "source_id": self.source_id,  "text": self.corpus, "match_words": []}
        blacklist = ["table. ", "tab. ", "fig. ", "figure. "]
        match_words["match_words"] = self.auto_match(entities)

        return match_words

    def auto_match(self, entities, lower_case=True):
        entities = list(set(entities)) ## Remove duplicates
        match_words = set()
        A = Automaton()
        for entity in entities:
            # Chinese-English compatible lowercase conversion, replace keyword.lower() with custom function custom_lower_fast(keyword)
            entity_key = utils.custom_lower_fast(entity) if lower_case else entity
            # Retrieve entity, output corresponding (subject, entity)
            A.add_word(entity_key, entity)
        A.make_automaton()  # Build automaton
        # Initialize match_raw: idx records text id, text_len records text length, unique_count records number of matched entities
        _text = utils.custom_lower_fast(self.corpus) if lower_case else self.corpus
        try:
            for end_index, entity in A.iter(_text):
                end_index += 1
                start_index = end_index - len(entity)
                # If detected boundary is not a word boundary, skip
                if utils.is_word_boundary(_text, start_index, end_index):
                    match_words.add(entity)
        except Exception as e:
            pass
        return list(match_words)
