from flask import Flask, request, jsonify
import index
from index import create_index, tiktoken_len # Import the create_index function
import qa
from qa import fetch_providers
import json

app = Flask(__name__)

@app.route('/create_index', methods=['POST'])
def create_index_route():
   data = request.get_json()
   csv_file_path = data['csv_file_path']
   result= create_index(csv_file_path)
   response = {
      'status' : 200,
      'message' : result
   }

   return jsonify(response)

@app.route('/get_providers', methods=['GET'])
def get_providers():
    query = request.args.get("query")

    # Fetch the 5 most suitable providers from your Pinecone index...
    # This will depend on your specific implementation
    results = fetch_providers(query)

    # Extract the top 3 results.
    top_results = results[:3]

    # Convert the results to JSON format.
    json_results = []
    for result in top_results:
       json_results.append({
       "id":result["id"]  ,
       "name": result["Name"],
       "designation": result["designation"],
       'field':result['field'],
       'phone':result['phone'],
       "location": result["location"],
       "description": result["description"],
       'School':result['School'],
       'programme':result['programme'],
       'duration':result['duration'],
       'company':result['company'],
       'position':result['position'],
       'jd':result['JD'],
       'experience':result['experience'],
    })

    return json.dumps(json_results)


if __name__ == '__main__':
   app.run(debug=True)