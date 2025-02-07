import pandas as pd
from flask import Flask, request, jsonify
from utils import (
    data_list,
    keyword_match, 
    fuzzy_match, 
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

        hr_role, hr_id = data_list(data)

        # Sort the results based on fuzzy match
        similar_job_ids = fuzzy_match(user_query.lower(), hr_role, hr_id)
        data = sort_results(data, similar_job_ids)

        new_hr_role, new_hr_id = data_list(data)
        
        # Sort the results based on keyword match
        matched_ids = keyword_match(user_query.lower(), new_hr_role, new_hr_id)
        data = sort_results(data, matched_ids)

        result = data.to_dict(orient='records')

        return jsonify({"results": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)