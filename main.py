import pandas as pd
from flask import Flask, request, jsonify
from utils import (
    job_title_generator,
    data_list,
    keyword_match,
    similar_query, 
    sort_results, 
    loading_embeddings)

df = pd.read_excel(r'dataset\updated_excel_file.xlsx')

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def searching():
    try:
        request_data = request.get_json()
        user_query = request_data.get('query', '')
        
        if not user_query:
            return jsonify({"error": "Query parameter is required."}), 400

        # Find similar queries using vector database
        vector_db = loading_embeddings(df)
        data = similar_query(user_query, vector_db, 1702)

        # Extract Role and ID Data
        new_hr_role, new_hr_id = data_list(data)

        # Expand the user query
        query_expansion = job_title_generator(user_query)
        query_expansion = list(dict.fromkeys([user_query] + query_expansion))  # Ensure order & remove duplicates

        print('➡ query_expansion:', query_expansion)

        # Dictionary to track unique matches while preserving order
        sorted_results = []
        seen_ids = set()

        # Process each job title in query expansion
        for job in query_expansion:
            matched_ids = keyword_match(job.lower(), new_hr_role, new_hr_id)
            
            for job_id in matched_ids:
                if job_id not in seen_ids:
                    sorted_results.append(data[data["HR ID"] == job_id])  # Select matching row
                    seen_ids.add(job_id)

        # Concatenate results into a final DataFrame
        final_data = pd.concat(sorted_results, ignore_index=True)

        # Convert to dictionary format
        result = final_data.to_dict(orient='records')


        return jsonify({"results": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
