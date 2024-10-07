from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
import pandas as pd
from pymilvus import MilvusClient
import json
import ollama

from transformers import AutoTokenizer
import uuid
import time
import pickle
import redis


model_id = "meta-llama/Meta-Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tiktoken_len(text):
    tokens = tokenizer(
        text,
        return_tensors="pt"
    )["input_ids"][0]
    return len(tokens)

def generate_thread_id():
    return f"thread_{uuid.uuid4().hex}"

def generate_message_id():
    return uuid.uuid4().hex

class HybridRetriever:
    def __init__(self, uri, col_name, model, embedding_model, sparse_embedding_model, output_fields, limit=15):
        self.uri = uri
        self.col_name = col_name
        self.model = model
        self.dense_embedding_model = embedding_model
        self.sparse_embedding_model = sparse_embedding_model
        self.output_fields = output_fields
        self.limit = limit
        self.connect_milvus()
        self.col = Collection(self.col_name, consistency_level="Strong")
        self.client = MilvusClient(uri=self.uri)

    def connect_milvus(self):
        connections.connect(uri=self.uri)

    def kw_extractor(self, query, history):
        output = ollama.generate(
            model=self.model,
            prompt=f"""
                Question: 
                    {query}
                Chat History:
                    {history}
                System Instructions:
                - return me a string of key words/phrases from the given question
                - take into the consideration the chat history, if the given question somehow referes to it, 
                    then extract the keywords/phrases also from the history of chat only related ones.
                - Do not do references to the text in you answer 
                - Do not provide comments from your side.                
                """
        )
        res = output['response']
        print(res)
        return self.sparse_embedding_model([res])["sparse"][[0]], ollama.embeddings(model=self.dense_embedding_model, prompt=res)["embedding"]

    def hybrid_search(self, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0):
        dense_search_params = {"index_type": "GPU_IVF_FLAT", "metric_type": "IP", "field_name": "dense_vector", "params": {"nlist": 4096}}
        dense_req = AnnSearchRequest([query_dense_embedding], "dense_vector", dense_search_params, limit=self.limit)
        
        sparse_search_params = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP", "field_name": "sparse_vector"}
        sparse_req = AnnSearchRequest([query_sparse_embedding], "sparse_vector", sparse_search_params, limit=self.limit)
        
        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search([sparse_req, dense_req], rerank=rerank, limit=self.limit, output_fields=self.output_fields)[0]
        return res

    def convert_explode_order_and_sort(self, obj_list):
        df = pd.DataFrame([obj.__dict__ for obj in obj_list])
        
        # if 'fields' in df.columns:
        #     fields_df = pd.json_normalize(df['fields'])
        #     df = df.drop(columns=['fields']).join(fields_df)
            
        if 'fields' in df.columns:
            # Extract the metadata column first
            metadata = df['fields'].apply(lambda x: x.get('metadata', None))
            
            # Normalize the fields without the metadata
            fields_df = pd.json_normalize(df['fields'].apply(lambda x: {k: v for k, v in x.items() if k != 'metadata'}))
            
            # Drop the original fields column and join the normalized data and metadata
            df = df.drop(columns=['fields']).join(fields_df).assign(metadata=metadata)
            # print(df.columns)
        column_order = ['distance', "document_id", "chunk_id", "file_name", "chunk_name", "chunk_text", "chunk_token_length","metadata"]
        df = df[[col for col in column_order if col in df.columns]]
        df = df.sort_values(by='distance', ascending=True)
        return df

    def new_row_to_df(self, res):
        new_row = {
            'distance': None,
            'document_id': res['document_id'],
            'chunk_id': res['chunk_id'],
            'file_name': res['file_name'],
            'chunk_name': res['chunk_name'],
            'chunk_text': res['chunk_text'],
            'chunk_token_length': res['chunk_token_length']
        }
        return pd.DataFrame([new_row])

    def get_data_milvus(self, doc_id, chunk_id):
        try:
            res = self.client.query(
                collection_name=self.col_name,
                filter=f'(document_id == "{doc_id}") and (chunk_id == {chunk_id})',
                output_fields=self.output_fields,
                limit=1
            )
            return res[0] if res else None
        except Exception as e:
            print(f"Error retrieving chunk {chunk_id} for document {doc_id}: {e}")
            return None

    def get_final_text(self, query="", history="") -> str:
        sparse_embeddings, dense_embeddings = self.kw_extractor(query,history)
        
        hybrid_results = self.hybrid_search(
            dense_embeddings, sparse_embeddings, sparse_weight=0.5, dense_weight=0.5
        )

        df = self.convert_explode_order_and_sort(hybrid_results)
        json_result = df.to_json(orient='records')
        parsed_json = json.loads(json_result)
        # pretty_json = json.dumps(parsed_json, indent=4)
        
        min_distance_idx = df.groupby('document_id')['distance'].idxmin()
        df_min_distance = df.loc[min_distance_idx].reset_index(drop=True)
        df_min_distance_sorted = df_min_distance.sort_values(by='distance', ascending=True)
        
        result_list = list(df_min_distance_sorted[['document_id', 'distance']].itertuples(index=False, name=None))
        
        document_chunks = {}
        for document_id, _ in result_list:
            relevant_rows = df[df['document_id'] == document_id].sort_values(by='chunk_id')
            document_chunks[document_id] = relevant_rows['chunk_id'].tolist()

        for doc_id in document_chunks:
            for chunk_id in document_chunks[doc_id]:
                if chunk_id - 1 not in document_chunks[doc_id]:
                    prev_chunk_data = self.get_data_milvus(doc_id, chunk_id - 1)
                    if prev_chunk_data:
                        new_row_df = self.new_row_to_df(prev_chunk_data).dropna(axis=1, how='all')
                        if int(new_row_df["chunk_token_length"][0]) + int(df['chunk_token_length'].sum()) < 25000:
                            df = pd.concat([df, new_row_df], ignore_index=True)

                if chunk_id + 1 not in document_chunks[doc_id]:
                    next_chunk_data = self.get_data_milvus(doc_id, chunk_id + 1)
                    if next_chunk_data:
                        new_row_df = self.new_row_to_df(next_chunk_data).dropna(axis=1, how='all')
                        if int(new_row_df["chunk_token_length"][0]) + int(df['chunk_token_length'].sum()) < 25000:
                            df = pd.concat([df, new_row_df], ignore_index=True)


        df = df.drop_duplicates(subset=['document_id', 'chunk_id', 'file_name', 'chunk_name', 'chunk_text', 'chunk_token_length'])
        
        df_sorted = df.sort_values(by=['document_id', 'chunk_id'])
        
        concatenated_texts = []
        for document_id in result_list:
            group = df_sorted[df_sorted['document_id'] == document_id[0]]
            concatenated_text = " ".join(group['chunk_text'].tolist())
            concatenated_texts.append(concatenated_text)
        
        final_concatenated_text = "\n\n".join(concatenated_texts)
        return final_concatenated_text, parsed_json
