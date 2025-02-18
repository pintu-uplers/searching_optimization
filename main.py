import warnings
warnings.filterwarnings("ignore")
import custom_logs
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
        custom_logs.log_action("searching", f"============================New Request============================")
        request_data = request.get_json()

        if not request_data:
            custom_logs.log_action("searching", f"Request data is empty.")
            return jsonify({"error": "Request data is empty.", "expected_format": {"query": "", "category": ""}}), 400

        custom_logs.log_action("searching", f"Request data: {request_data}.")
        user_query = request_data.get('query', '')
        job_category = request_data.get('category', '')
        
        if not user_query:
            custom_logs.log_action("searching", f"query parameter is required.")
            return jsonify({"error": f"query parameter is required."}), 400
        
        if not job_category:
            custom_logs.log_action("searching", f"category parameter is required.")
            return jsonify({"error": f"category parameter is required."}), 400

        vector_db = loading_embeddings()

        # Find similar queries using vector database
        data = similar_query(user_query, job_category, vector_db, 5)

        data = sort_results(user_query, data)
    
        # Convert to dictionary format
        custom_logs.log_action("searching", f"Results found: {len(data)}")
        result = data.to_dict(orient='records')


        return jsonify({"results": result}), 200
    except Exception as e:
        # custom_logs.log_action("searching", f"Error in searching: {e}", "error")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
