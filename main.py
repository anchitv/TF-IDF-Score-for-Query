import spacy
from collections import defaultdict
import math
import copy


class InvertedIndex:
    def __init__(self):
        self.tf_tokens = defaultdict(lambda: defaultdict(int))
        self.tf_entities = defaultdict(lambda: defaultdict(int))
        self.idf_tokens = defaultdict()
        self.idf_entities = defaultdict()


    @staticmethod
    def idf_formula(tot_doc, num_docs):
        return 1 + math.log((tot_doc / (1 + num_docs)))

    @staticmethod
    def tf_idf(tf, idf, descriptor):
        if tf == 0:
            return 0
        if descriptor == 'token':
            tf = 1 + math.log(tf)

        return (1 + math.log(tf)) * idf

    def index_documents(self, documents):
        nlp = spacy.load("en_core_web_sm")
        total_doc = len(documents)

        for k,v in documents.items():
            doc = nlp(v)
            
            # TF of token
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    self.tf_tokens[token.text][k] += 1
            
            # # TF of entity
            for entity in doc.ents:
                ent = entity.text
                if self.tf_tokens[ent]:
                    self.tf_tokens[ent][k] -= 1
                    if not self.tf_tokens[ent][k]:
                        del self.tf_tokens[ent][k]
                    if not self.tf_tokens[ent]:
                        del self.tf_tokens[ent]
                self.tf_entities[ent][k] += 1

        self.tf_tokens = {k:v for k,v in self.tf_tokens.items() if v}

        # IDF of term and normalisation of TF
        for token, doc in self.tf_tokens.items():
            self.idf_tokens[token] = self.idf_formula(total_doc, len(doc))

        # IDF of entity and normalisation of TF
        for entity, doc in self.tf_entities.items():
            self.idf_entities[entity] = self.idf_formula(total_doc, len(doc))

        return 1
        

    @staticmethod
    def token_maker(entities, query):
        # Function to separate given entities from query and return
        # a list of remaining tokens.
        # 
        # entities - A list of entities, each entity containing space separated tokens
        # query - A string consisting tokens separated by spaces.

        for entity in entities:
            for token in entity.split(' '):
                query = query.replace(token, "", 1)

        return query.split()
        
    @staticmethod
    def check_entity(entity, Q):
        # Check if an entity is present in Q. The tokens in an entity have to be present
        # in the same order in Q but can have other tokens in between them.
        # 
        # entity - An entity consisting of one or more space separated tokens
        # Q - A string consisting of tokens separated by spaces.

        query = Q.split()
        i=0
        for token in entity.split():
            if token in query:
                query = query[query.index(token)+1:]
            else:
                i=1
                break

        if i==1:
            return False
        
        return ' '.join(InvertedIndex.token_maker([entity], Q))

    @staticmethod
    def find_entities(query, entity_list):
        # Returns a list of entities found in the query from
        # a given list of entities.
        # 
        # query - A string consisting of tokens separated by spaces.
        # entity_list - A list of possible entities in query.

        entity_dict = {}
        for entity in entity_list:
            x = InvertedIndex.check_entity(entity, query)
            if x != False:
                entity_dict[entity] = x
        
        entity_list.clear()
        entity_list += list(entity_dict.keys())

        return entity_dict

    @staticmethod
    def append_dict(entities, split_list, Q):
        # Appends a dictionary of entities and remaining tokens
        # in Q in split_list.
        # 
        # entities - A list of entities.
        # split_list - A list of dictionaries containing entities
        #           and tokens. Output of split_query function.
        # Q - The main query input by user.

        for lis in split_list:
            if set(entities) == set(lis['entities']):
                return False
        split_list.append({'entities': entities, 'tokens': InvertedIndex.token_maker(entities, Q)})

        return True

    @staticmethod
    def permute(entity_list, query, Q, split_list):
        # Returns a list of all possible combinations of entities and
        # remaining tokens in Q. Each combination is represented in a
        # dictionary with 'entities' and 'tokens' as keys with values as
        # a list of the entities and tokens respectively.
        # 
        # entity_list - A list of possible entities in query.
        # query - The query with some entities removed from Q.
        # Q - The complete Query given by user. Used to get tokens after removing entities.
        # split_list - A list of dictionaries which is returned.

        if not entity_list:
            return

        in_dict = InvertedIndex.find_entities(query, entity_list)

        if not in_dict:
            return

        for k,v in in_dict.items():
            x = copy.deepcopy(entity_list)
            x.remove(k)
            temp_list = []
            InvertedIndex.permute(x, v, Q, temp_list)
            
            for lis in temp_list:
                if k not in lis['entities']:
                    entity = lis['entities']+[k]
                    InvertedIndex.append_dict(entity, split_list, Q)
            InvertedIndex.append_dict([k], split_list, Q)
                    
        return

    def split_query(self, Q, DoE):
        entity_list = list(DoE.keys())
        split_list = []
        self.permute(entity_list, Q, Q, split_list)
        split_list += [{'entities': [], 'tokens': InvertedIndex.token_maker([], Q)}]

        return split_list


    def max_score_query(self, query_splits, doc_id):
        max_score = 0

        for query in query_splits:
            s1 = 0
            s2 = 0
            for token in query['tokens']:
                if token in self.tf_tokens.keys():
                    s1 += self.tf_idf(self.tf_tokens[token][doc_id], self.idf_tokens[token], 'token')
            for entity in query['entities']:
                if entity in self.tf_entities.keys():
                    s2 += self.tf_idf(self.tf_entities[entity][doc_id], self.idf_entities[entity], 'entity')
            
            score = s2+0.4*s1
            print('query = ', 'token ', s1, ' entity ', s2, ' total ', score, query, '\n')
            
            if score > max_score:
                max_score = score
                output = max_score, query

        return output 
        
