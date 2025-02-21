import os
import string
import custom_logs
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from langchain.schema import Document
from flashtext import KeywordProcessor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from traceback import format_exc
from dotenv import load_dotenv
load_dotenv()


def fuzzy_match(user_role, candidate_role):
    try:
        punctuation_to_remove = string.punctuation.replace('#', '').replace('+', '').replace('-', '')

        queries = user_role.lower().split()
        candidate_role_token = candidate_role.lower().split()
        
        # Remove unwanted punctuation from job profile tokens
        clean_candidate_role = [
            ''.join(char for char in word if char not in punctuation_to_remove)
            for word in candidate_role_token
        ]

        matches = [word for word in clean_candidate_role if max(fuzz.partial_ratio(word, query) for query in queries) > 80]

        if matches:
            custom_logs.log_action("fuzzy_match", f"Matches found.")
            job_role = max(
                (fuzz.ratio(user_role.lower(), match) for match in matches), 
                default=0)
            
            job_title = ' '.join(matches)
            job_role_ratio = fuzz.ratio(user_role.lower(), job_title)

            max_similarity = max((job_role, job_role_ratio), default=0)
            return max_similarity
        return 0

    except Exception as e:
        print(format_exc())
        custom_logs.log_action("fuzzy_match", f"Error in fuzzy match: {e}", log_level="error")
        return []

def match_criteria(metadata, criteria, threshold=80):
    try:
        if "Role" in criteria and "Role" in metadata:
            # custom_logs.log_action("match_criteria", f"Matching {criteria["Role"]} with {metadata["Role"]}.")
            score = fuzzy_match(str(criteria["Role"]).lower(), str(metadata["Role"]).lower())
            if score is not None and isinstance(score, (int, float)) and score >= threshold:
                custom_logs.log_action("match_criteria", f"Matched Role.")
                return True
        return False
    except Exception as e:
        custom_logs.log_action("match_criteria", f"Error in matching criteria: {e}", log_level="error")


def keyword_match(user_query, data):
    try:
        role_description = list(data['Role'])
        talent_id = list(data['ID'])
        matched_ids = []
        keywordprocessor = KeywordProcessor(case_sensitive=False)
        keywordprocessor.add_keyword(user_query)  # Add the user query as a keyword

        for job_profile, idx in zip(role_description, talent_id):
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
        custom_logs.log_action("loading_embeddings", f"Folder exists.")
        os.makedirs(save_folder, exist_ok=True)
        faiss_index_path = os.path.join(save_folder, faiss_index_file)

        if os.path.exists(faiss_index_path):
            # print(f"FAISS index loading from {faiss_index_path}.")
            custom_logs.log_action("loading_embeddings", f"FAISS Database loading from {faiss_index_path}.")

            vector_db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)

            return vector_db
        
        # print(f"Creating FAISS index {faiss_index_path}.")
        custom_logs.log_action("loading_embeddings", f"Creating FAISS index {faiss_index_path}.")
        df = pd.read_excel(r'dataset\updated_resume.xlsx')
        role_description = df["job_description"].fillna("").tolist()
        talent_id = df["talent_id"].fillna("").tolist()
        role_title = df["title"].fillna("").tolist()
        experience = df["experience"].fillna("").tolist()
        company = df["company_name"].fillna("").tolist()

        documents = [
            Document(page_content=desc, metadata={"ID": idx, "Role": title, "Company":comp, "Experience": round(exp/12)})
            for desc, idx, title, comp, exp in zip(role_description, talent_id, role_title, company, experience)
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


def similar_query(validated_data, vector_db, k):
    try:
        custom_logs.log_action("similar_query", f"Performing similarity search for {validated_data}.")

        # user_query = request_data.get('query', '')

        df_results = getting_results(validated_data, vector_db, k)
        
        
        return df_results
    except Exception as e:
        custom_logs.log_action("similar_query", f"Error in similar query: {format_exc()}", log_level="error")


def sort_results(user_query, data):
    """ Sort the results based on match """
    try:
        matched_ids = keyword_match(user_query, data)
        data['priority_order'] = data['ID'].apply(lambda x: matched_ids.index(x) if x in matched_ids else float('inf'))
        data['original_index'] = data.index
        data = data.sort_values(by=['priority_order', 'original_index']).drop(columns=['priority_order', 'original_index']).reset_index(drop=True)

        custom_logs.log_action("sort_results", f"Results sorted: {len(data)}")
        return data
    except Exception as e:
        custom_logs.log_action("sort_results", f"Error in sorting results: {e}", log_level="error")

def df_creation(results, file_name, output_dir="output"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_name}.xlsx")

        data = [
            {
                "ID": str(doc.metadata.get("ID", "Unknown")),
                "Role": str(doc.metadata.get("Role", "Unknown")),
                "Experience": str(doc.metadata.get("Experience", "Unknown")),
            }
            for doc in results
        ]
        df_results = pd.DataFrame(data)
        df_results.to_excel(f"{output_path}", index=False)
        return df_results
    except Exception as e:
        custom_logs.log_action("df_creation", f"Error in creating dataframe: {e}", log_level="error")
        return pd.DataFrame(columns=["ID", "Role", "Experience"])
    

def getting_results(validated_data, vector_db, k):
    try:
        
        user_query = validated_data.query

        criteria = {
                "Role": validated_data.role,
                "Experience": validated_data.exp_in_years,
                "Company": validated_data.company,
            }
        
        if user_query and (validated_data.exp_in_years is None and validated_data.role is None):
            custom_logs.log_action("getting_results", f"Searching for similar queries.")
            results_vector = vector_db.similarity_search(user_query, k=k)
            df_results = df_creation(results_vector, "vector_output")
        elif validated_data.exp_in_years is not None or validated_data.role is not None:
            custom_logs.log_action("getting_results", f"Searching for similar queries with criteria.")
            # results = vector_db.similarity_search(user_query, k=k, filter={"Experience": validated_data.exp_in_year})
            results_vector = vector_db.similarity_search(user_query, k=k)
            # print('➡ results_vector:', results_vector)
            df_results = df_creation(results_vector, "vector_output")

            custom_logs.log_action("getting_results", f"All Results found: {len(results_vector)}")

            results_exp = [
                res for res in results_vector 
                if res.metadata.get("Experience") is not None and isinstance(res.metadata["Experience"], (int, float)) 
                and res.metadata["Experience"] <= validated_data.exp_in_years
            ]

            df_results = df_creation(results_exp, "exp_output")

            custom_logs.log_action("getting_results", f"Results found from Exp filter: {len(results_exp)}")
            
            results_role = [doc for doc in results_vector if match_criteria(doc.metadata, criteria)]
            df_results = df_creation(results_role, "role_output")
            custom_logs.log_action("getting_results", f"Results found from Role: {len(results_role)}")

        if not results_vector:
            custom_logs.log_action("getting_results", f"No results found.")
            return pd.DataFrame(columns=["ID", "Role", "Experience"])

        return df_results

    except Exception as e:
        custom_logs.log_action("getting_results", f"Error in getting results: {e}", log_level="error")
        return pd.DataFrame(columns=["ID", "Role", "Experience"])