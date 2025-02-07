import os
import string
import pandas as pd
from rapidfuzz import fuzz
from langchain.schema import Document
from flashtext import KeywordProcessor
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def data_list(data):
    try:
        hr_role = list(data['Job Title'])
        hr_id = list(data['HR ID'])
        return hr_role, hr_id
    except Exception as e:
        print(e)

def keyword_match(user_query, hr_role, hr_id):
    try:
        matched_ids = []
        keywordprocessor = KeywordProcessor(case_sensitive=False)
        keywordprocessor.add_keyword(user_query)  # Add the user query as a keyword

        for job_profile, idx in zip(hr_role, hr_id):
            extracted_keywords = keywordprocessor.extract_keywords(job_profile.lower())
            # Append id only if keywords are extracted
            if extracted_keywords:
                matched_ids.append(idx)
        
        return matched_ids

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def fuzzy_match(user_query, hr_role, hr_id):
    try:
        similar_jobs = []

        queries = user_query.lower().split()
        punctuation_to_remove = string.punctuation.replace('#', '').replace('+', '').replace('-', '')
        
        for job_profile, idx in zip(hr_role, hr_id):
            job_profile_token = job_profile.lower().split()
            
            # Remove unwanted punctuation from job profile tokens
            filtered_job_profile = [
                ''.join(char for char in word if char not in punctuation_to_remove)
                for word in job_profile_token
            ]
            filtered_job_profile = [word for word in filtered_job_profile if word]

            matches = [word for word in filtered_job_profile if max(fuzz.partial_ratio(word, query) for query in queries) > 80]

            if matches:
                job_title_keyword_ratio = max(
                    (fuzz.ratio(user_query.lower(), match) for match in matches), 
                    default=0
                )
                job_title = ' '.join(matches)
                job_title_ratio = fuzz.ratio(user_query.lower(), job_title)

                max_similarity = max(job_title_keyword_ratio, job_title_ratio)

                if max_similarity > 80:
                    similar_jobs.append(idx)
        return similar_jobs

    except Exception as e:
        print('➡ error in fuzzy match:', e)
        return []


def loading_embeddings(df, faiss_index_file="faiss_index", save_folder="faiss_indices_db"):
    try:
        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)
        faiss_index_path = os.path.join(save_folder, faiss_index_file)

        if os.path.exists(faiss_index_path):
            print(f"FAISS index loading from {faiss_index_path}.")

            vector_db = FAISS.load_local(faiss_index_path, OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)

            return vector_db
        
        hr_role = df["hr_role"].fillna("").tolist()
        hr_id = df["HR_Number"].fillna("").tolist()

        documents = [
            Document(page_content=role, metadata={"hr_id": idx})
            for role, idx in zip(hr_role, hr_id)
        ]
        # Generate embeddings and create FAISS vector store
        vector_db = FAISS.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-3-large")
        )
        # Save the FAISS index to the specified folder
        vector_db.save_local(faiss_index_path)

        # vector_db.similarity_search
        print(f"FAISS index saved at {faiss_index_path}.")
        
        return vector_db
    except Exception as e:
        print("Error in loading embeddings", e)


def similar_query(user_query, vector_db, k):
    try:
        results = vector_db.similarity_search(user_query, k=k)

        data = [
            {
                "HR ID": str(doc.metadata.get("hr_id")),
                "Job Title": str(doc.page_content)
            }
            for doc in results
        ]

        df_results = pd.DataFrame(data)
        
        return df_results
    except Exception as e:
        print("Error in similar query", e)


def sort_results(data, matched_ids):
    """ Sort the results based on match """
    try:
        data['priority_order'] = data['HR ID'].apply(lambda x: matched_ids.index(x) if x in matched_ids else float('inf'))
        data['original_index'] = data.index
        data = data.sort_values(by=['priority_order', 'original_index']).drop(columns=['priority_order', 'original_index']).reset_index(drop=True)

        return data
    except Exception as e:
        print(e)


