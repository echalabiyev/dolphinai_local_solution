import streamlit as st
import pandas as pd
import time
from milvus_model.hybrid import BGEM3EmbeddingFunction
from pymilvus import (
    connections,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
    MilvusClient,
)
import ollama


class RetrieveAugmentGenerate:
    def __init__(self, uri, ef, model="llama3.1:8b", col_name="hybrid_sap_collection_llama_7b", limit=10):
        self.uri = uri
        self.model = model
        self.col_name = col_name
        self.limit = limit
        self.output_fields = ["document_id", "chunk_id", "file_name", "chunk_name", "chunk_text", "chunk_token_length"]

        # Establish connection
        connections.connect(uri=uri)
        self.col = Collection(self.col_name, consistency_level="Strong")
        self.ef = ef  # Pass the embedding function instance
        self.client = MilvusClient(uri=uri)
        self.dense_dim = self.ef.dim["dense"]

    def hybrid_search(self, query_dense_embedding, query_sparse_embedding, sparse_weight=1.0, dense_weight=1.0):
        dense_search_params = {
            "index_type": "GPU_IVF_FLAT",
            "metric_type": "IP",
            "field_name": "dense_vector",
            "params": {"nlist": 8192}
        }
        dense_req = AnnSearchRequest([query_dense_embedding], "dense_vector", dense_search_params, limit=self.limit)

        sparse_search_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "field_name": "sparse_vector"
        }
        sparse_req = AnnSearchRequest([query_sparse_embedding], "sparse_vector", sparse_search_params, limit=self.limit)

        rerank = WeightedRanker(sparse_weight, dense_weight)
        res = self.col.hybrid_search(
            [sparse_req, dense_req],
            rerank=rerank,
            limit=self.limit,
            output_fields=self.output_fields
        )[0]
        return res

    def convert_explode_order_and_sort(self, obj_list):
        df = pd.DataFrame([obj.__dict__ for obj in obj_list])
        if 'fields' in df.columns:
            fields_df = pd.json_normalize(df['fields'])
            df = df.drop(columns=['fields']).join(fields_df)
        
        column_order = ['distance', "document_id", "chunk_id", "file_name", "chunk_name", "chunk_text", "chunk_token_length"]
        df = df[[col for col in column_order if col in df.columns]]
        df = df.sort_values(by='distance', ascending=True)
        return df

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

    def retrieve_augmented_text(self, query):
        kw_string = ollama.generate(
            model=self.model,
            prompt=f"""
                Question: 
                    {query}
                System Instructions:
                - Return a string of key words/phrases from the given question
                - Do not do references to the text in your answer 
                - Do not provide comments from your side.                
            """
        )['response']

        query_embeddings = self.ef([kw_string])
        query_dense_embeddings = ollama.embeddings(model=self.model, prompt=kw_string)["embedding"]

        hybrid_results = self.hybrid_search(query_dense_embeddings, query_embeddings["sparse"][[0]])

        df = self.convert_explode_order_and_sort(hybrid_results)
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
                        df = pd.concat([df, pd.DataFrame([prev_chunk_data])], ignore_index=True)

                if chunk_id + 1 not in document_chunks[doc_id]:
                    next_chunk_data = self.get_data_milvus(doc_id, chunk_id + 1)
                    if next_chunk_data:
                        df = pd.concat([df, pd.DataFrame([next_chunk_data])], ignore_index=True)

        df = df.drop_duplicates(subset=['document_id', 'chunk_id', 'file_name', 'chunk_name', 'chunk_text', 'chunk_token_length'])
        df_sorted = df.sort_values(by=['document_id', 'chunk_id'])

        concatenated_texts = []
        for document_id in result_list:
            group = df_sorted[df_sorted['document_id'] == document_id[0]]
            concatenated_text = " ".join(group['chunk_text'].tolist())
            concatenated_text = f"\n\nFile name: {df_sorted[df_sorted['document_id'] == document_id[0]].reset_index()["file_name"][0]}\n" + concatenated_text.strip()
            print(concatenated_text)
            concatenated_texts.append(concatenated_text)

        final_concatenated_text = "\n\n".join(concatenated_texts)

        # Extract unique file names from the final dataframe
        unique_file_names = df_sorted['file_name'].unique().tolist()

        return final_concatenated_text, unique_file_names

    def generate_response(self, query):
        augmented_text, unique_file_names = self.retrieve_augmented_text(query)
        output = ollama.generate(
            model=self.model,
            prompt=f"""Using this data: {augmented_text}. Provide a comprehensive answer to this prompt: {query}.
                System Instructions:
                - Do not do references to the text in your answer 
                - Do not provide comments from your side.
                """
        )
        return output['response'], unique_file_names


# Streamlit application
def main():
    st.title("DolphinAI")

    uri = "http://localhost:19530/dolphinai_db"
    default_model = "llama3.1:8b"
    default_collection = "hybrid_sap_collection_llama_7b"
    default_limit = 10

    # Initialize the BGEM3EmbeddingFunction when the app starts
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda:1")

    # Moving the settings to the sidebar
    with st.sidebar:
        st.title("Settings")
        # Dropdown options for model and collection names
        model = st.selectbox("Model Name:", ["llama3.1:8b", "dolphinai-mixtral:8x7b"], index=0)
        col_name = st.selectbox("Collection Name:", ["hybrid_sap_collection_llama_7b", "hybrid_transactions_collection_llama_7b"], index=0)
        limit = st.number_input("Limit:", min_value=1, value=default_limit)

    query = st.text_area("Enter your query:")

    if st.button("Submit Query"):
        start_time = time.time()  # Start the timer
        rag = RetrieveAugmentGenerate(uri, ef, model=model, col_name=col_name, limit=limit)
        response, unique_file_names = rag.generate_response(query)
        end_time = time.time()  # End the timer

        # Display response and the time taken
        st.markdown(response, unsafe_allow_html=True)
        st.write("Reference Files:", unique_file_names)
        st.write(f"Time taken to receive the answer: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()

