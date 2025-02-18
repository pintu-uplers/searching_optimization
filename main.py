import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from flask import Flask, request, jsonify
from utils import (
    similar_query, 
    loading_embeddings,
    sort_results)

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def searching():
    try:
        request_data = request.get_json()

        if not request_data:
            return jsonify({"error": "Request data is empty.", "expected_format": {"query": "", "category": ""}}), 400

        user_query = request_data.get('query', '')
        job_category = request_data.get('category', '')
        
        if not user_query:
            return jsonify({"error": f"query parameter is required."}), 400
        
        if not job_category:
            return jsonify({"error": f"category parameter is required."}), 400

        vector_db = loading_embeddings()

        # Find similar queries using vector database
        data = similar_query(user_query, job_category, vector_db, 5)

        data = sort_results(user_query, data)
    
        # Convert to dictionary format
        result = data.to_dict(orient='records')


        return jsonify({"results": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
