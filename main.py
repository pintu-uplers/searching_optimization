import warnings
import time
from traceback import format_exc
import custom_logs
from schema import RequestSchema
warnings.filterwarnings("ignore")
from flask import Flask, request, jsonify
from utils import (
    similar_query, 
    loading_embeddings)

app = Flask(__name__)

@app.route('/search', methods=['POST'])
def searching():
    try:
        start_time = time.time()
        custom_logs.log_action("searching", f"============================New Request============================")
        request_data = request.get_json()

        if not request_data:
            # custom_logs.log_action("searching", f"Request data is empty.")
            return jsonify({"error": "Request data is empty.", "expected_format": {"query": "", "role": ""}}), 400
        
        validated_data = RequestSchema(**request_data)

        custom_logs.log_action("searching", f"Request data: {request_data}.")

        vector_db = loading_embeddings()

        # Find similar queries using vector database
        data = similar_query(validated_data, vector_db, 100)

        data = data[["Role", "Experience", "Current", "Cosine_distance", "Final_Score"]]
    
        # Convert to dictionary format
        custom_logs.log_action("searching", f"Result displayed")
        result = data.to_dict(orient='records')

        end_time = time.time()
        custom_logs.log_action("searching", f"Time taken: {round(end_time - start_time, 2)} seconds.")
        custom_logs.log_action("searching", f"============================End Request============================")

        return jsonify({"results": result}), 200
    except Exception as e:
        custom_logs.log_action("searching", f"Error in searching: {format_exc()}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
