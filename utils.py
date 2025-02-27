import os
import custom_logs
import numpy as np
import pandas as pd
from datetime import datetime
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

from traceback import format_exc
from dotenv import load_dotenv
load_dotenv()

def loading_embeddings():
    try:
        persist_directory  = "./chroma_db"

        if os.path.exists(persist_directory):

            custom_logs.log_action("loading_embeddings", f"Chroma Database loading from {persist_directory}.")
            vector_db = Chroma(persist_directory=persist_directory,
                               embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
            
            return vector_db
        
        # print(f"Creating FAISS index {faiss_index_path}.")
        custom_logs.log_action("loading_embeddings", f"Creating Chroma index.")

        df = pd.read_excel(r'dataset\updated_resume.xlsx')

        role_description = df["job_description"].fillna("").tolist()
        talent_id = df["talent_id"].fillna("").tolist()
        role_title = df["title"].fillna("").tolist()
        experience = df["experience"].fillna("").tolist()
        company = df["company_name"].fillna("").tolist()
        current_job = df["is_current"].fillna("").tolist()
        start_date = df["start_date"].fillna("").tolist()
        end_date = df["end_date"].fillna("").tolist()

        documents = [
            Document(page_content=desc, metadata={"ID": idx, "Role": title, "Company":comp, "Experience": round(exp/12), "Current": current, "Start": start, "End": end})
            for desc, idx, title, comp, exp, current, start, end in zip(role_description, talent_id, role_title, company, experience, current_job, start_date, end_date)
        ]
        
        vector_db = Chroma.from_documents(
                        documents=documents,
                        embedding=OpenAIEmbeddings(model="text-embedding-3-large"),
                        persist_directory=persist_directory)
            
        return vector_db
    except Exception as e:
        custom_logs.log_action("loading_embeddings", f"Error in loading embeddings: {e}", log_level="error")


def df_creation(results, file_name, output_dir="output"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{file_name}.xlsx")

        data = [
            {
                "ID": str(doc.metadata.get("ID", "Unknown")),
                "Role": str(doc.metadata.get("Role", "Unknown")),
                "Cosine_distance": str(1 - score/2),
                "Experience": str(doc.metadata.get("Experience", "Unknown")),
                "Start": str(doc.metadata.get("Start", "Unknown")),
                "End": str(doc.metadata.get("End", "Unknown")),
                "Current": str(doc.metadata.get("Current", "Unknown")),
            }
            for doc, score in results
        ]
        df_results = pd.DataFrame(data)
        df_results.replace("", np.nan, inplace=True)

        custom_logs.log_action("df_creation", f"Top k fetch from vector db.")
        df_results.to_excel(f"{output_path}", index=False)
        return df_results
    except Exception as e:
        custom_logs.log_action("df_creation", f"Error in creating dataframe: {e}", log_level="error")
        return pd.DataFrame(columns=["ID", "Role", "Experience"]) 

def compute_recency_weight(start_date, end_date, current):
    try:
        now = datetime.now()
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')
        
        if pd.isna(start_date) and pd.isna(end_date) and current == 0:
            return 0.1
        
        if current == 1:
            end = end_date if not pd.isnull(end_date) else now
        else:
            end = end_date if not pd.isnull(end_date) else start_date

        end = pd.to_datetime(end, errors='coerce')
        years_ago = (now - end).days / 365.0
        years_ago = max(0, years_ago)

        lambda_ = 0.5
        weight = np.exp(-lambda_ * years_ago)

        return max(weight, 0.01)
    except Exception as e:
        custom_logs.log_action("compute_recency_weight", f"Error in compute_recency_weight results: {e}", log_level="error")


def getting_results(validated_data, vector_db, k):
    try:
        
        user_query = validated_data.query
        
        if user_query:
            custom_logs.log_action("getting_results", f"Searching for similar queries.")

            results_vector = vector_db.similarity_search_with_score(user_query, k=k)
            df_results = df_creation(results_vector, "vector_output")

        weights = []
        
        df_results["Current"] = pd.to_numeric(df_results["Current"], errors="coerce").astype(int)
        df_results["Cosine_distance"] = pd.to_numeric(df_results["Cosine_distance"], errors="coerce").astype(float)

        df_results.to_excel(r"local_files_dirs\Original_score.xlsx", index=False)
        custom_logs.log_action("compute_recency_weight", f"Converting datetime into %Y%M%D format")
        for start, end, current in zip(df_results["Start"], df_results["End"], df_results["Current"]):
            weight = compute_recency_weight(start, end, current)
            weights.append(weight)

        df_results['Recency_Weight'] = pd.Series(weights)

        custom_logs.log_action("getting_results", f"Updating weight based on new recency weight")
        df_results["Final_Score"] = df_results["Cosine_distance"] * df_results["Recency_Weight"]
        
        df_sorted = df_results.sort_values(by=["Final_Score"], ascending=[False])
        df_sorted.to_excel(r"local_files_dirs\Updated_score.xlsx", index=False)

        if not results_vector:
            custom_logs.log_action("getting_results", f"No results found.")
            return pd.DataFrame(columns=["ID", "Role", "Experience"])

        return df_sorted

    except Exception as e:
        custom_logs.log_action("getting_results", f"Error in getting results: {e}", log_level="error")
        return pd.DataFrame(columns=["ID", "Role", "Experience"])
    

def similar_query(validated_data, vector_db, k):
    try:
        custom_logs.log_action("similar_query", f"Performing similarity search for {validated_data}.")
        df_results = getting_results(validated_data, vector_db, k)
        return df_results
    except Exception as e:
        custom_logs.log_action("similar_query", f"Error in similar query: {format_exc()}", log_level="error")


