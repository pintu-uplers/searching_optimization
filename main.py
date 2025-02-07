import pandas as pd
from flask import Flask, request, jsonify
from fuzzy_match import fuzzy_match

df = pd.read_excel(r'dataset\updated_excel_file.xlsx')
hr_role = list(df['hr_role'])
hr_id = list(df['HR_Number'])

app = Flask(__name__)

@app.route('/spelling_correction', methods=['POST'])
def search():
    try:
        user_query = request.json['user_query']
        data = fuzzy_match(user_query, hr_role, hr_id)
        result = data.to_dict(orient='records')
        return jsonify({"results": result}), 200
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)