import os
import string
import json
import custom_logs
from groq import Groq
import prompt_template
import pandas as pd
from rapidfuzz import fuzz
from langchain.schema import Document
from flashtext import KeywordProcessor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


def keyword_match(user_query, data):
    try:
        custom_logs.log_action("keyword_match", f"Adding user query as keyword: {user_query}.")
        hr_role = list(data['Job Title'])
        hr_id = list(data['HR ID'])
        matched_ids = []
        keywordprocessor = KeywordProcessor(case_sensitive=False)
        keywordprocessor.add_keyword(user_query)  # Add the user query as a keyword

        for job_profile, idx in zip(hr_role, hr_id):
            extracted_keywords = keywordprocessor.extract_keywords(job_profile.lower())
            # Append id only if keywords are extracted
            if extracted_keywords:
                matched_ids.append(idx)
        
        custom_logs.log_action("keyword_match", f"Matched IDs: {matched_ids}.")
        
        return matched_ids

    except Exception as e:
        custom_logs.log_action("keyword_match", f"An error occurred: {e}", log_level="error")
        print(f"An error occurred: {e}")
        return []


def loading_embeddings(faiss_index_file="faiss_index", save_folder="faiss_indices_db"):
    try:
        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)
        faiss_index_path = os.path.join(save_folder, faiss_index_file)

        if os.path.exists(faiss_index_path):
            # print(f"FAISS index loading from {faiss_index_path}.")
            custom_logs.log_action("loading_embeddings", f"FAISS index loading from {faiss_index_path}.")

            vector_db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)

            return vector_db
        
        # print(f"Creating FAISS index {faiss_index_path}.")
        custom_logs.log_action("loading_embeddings", f"Creating FAISS index {faiss_index_path}.")
        df = pd.read_excel(r'dataset\Resume_HR.xlsx', sheet_name='hr_roles')
        hr_role = df["Role"].fillna("").tolist()
        hr_id = df["ID"].fillna("").tolist()
        category = df["Category"].fillna("").tolist()

        documents = [
            Document(page_content=role, metadata={"ID": idx, "Category": cat})
            for role, idx, cat in zip(hr_role, hr_id, category)
        ]
        # Generate embeddings and create FAISS vector store
        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        # Save the FAISS index to the specified folder
        vector_db.save_local(faiss_index_path)

        # print(f"FAISS index saved at {faiss_index_path}.")
        custom_logs.log_action("loading_embeddings", f"FAISS index saved at {faiss_index_path}.")
        
        return vector_db
    except Exception as e:
        custom_logs.log_action("loading_embeddings", f"Error in loading embeddings: {e}", log_level="error")
        # print("Error in loading embeddings", e)


def similar_query(user_query, job_category, vector_db, k):
    try:
        custom_logs.log_action("similar_query", f"User query: {user_query}, Job category: {job_category}, K: {k}.")
        custom_logs.log_action("similar_query", f"Performing similarity search.")
        results = vector_db.similarity_search(user_query, k=k, filter={"Category": job_category})
        # print('➡ results:', results)

        if not results:
            custom_logs.log_action("similar_query", f"No results found.")
            return pd.DataFrame(columns=["HR ID", "Category", "Job Title"])

        data = [
            {
                "HR ID": str(doc.metadata.get("ID", "Unknown")),
                "Category": str(doc.metadata.get("Category", "Unknown")),
                "Job Title": str(doc.page_content)
            }
            for doc in results
        ]

        df_results = pd.DataFrame(data)
        custom_logs.log_action("similar_query", f"Results found: {len(df_results)}")
        
        return df_results
    except Exception as e:
        # print("Error in similar query", e)
        custom_logs.log_action("similar_query", f"Error in similar query: {e}", log_level="error")


def sort_results(user_query, data):
    """ Sort the results based on match """
    try:
        matched_ids = keyword_match(user_query, data)
        data['priority_order'] = data['HR ID'].apply(lambda x: matched_ids.index(x) if x in matched_ids else float('inf'))
        data['original_index'] = data.index
        data = data.sort_values(by=['priority_order', 'original_index']).drop(columns=['priority_order', 'original_index']).reset_index(drop=True)

        custom_logs.log_action("sort_results", f"Results sorted.")
        return data
    except Exception as e:
        custom_logs.log_action("sort_results", f"Error in sorting results: {e}", log_level="error")
        print(e)


