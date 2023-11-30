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
       "id": result["id"],
      "Name": result["Name"],
      "phone": result["phone"],
      "headline": result["headline"],
      "isApproved": result["isApproved"],
      "address": {
                "country": result["address"]["country"],
                "city": result["address"]["city"],
                "state": result["address"]["state"],
                "zipCode": result["address"]["zipCode"],
                "addressLineOne": result["address"]["addressLineOne"],
                "addressLineTwo": result["address"]["addressLineTwo"],
            },
      "services":{
        'services_id': result["services"]['services_id'],
            'services_name': result["services"]['services_name'],
            'services_desc': result["services"]['services_desc'],
      },
      "education":{
        "School": result["education"]["School"],
        "programme": result["education"]["programme"],
        "duration": result["education"]["duration"],
        'education_description':result["education"]['education_description'],
      },
      "workExperience": {
        "companyName": result["workExperience"]["companyName"],
        "position": result["workExperience"]["position"],
        "experience": result["workExperience"]["experience"],
        "JD": result["workExperience"]["JD"],
      },
      "achievements_description": result["achievements_description"],
      "about": result["about"],
    })

    return json.dumps(json_results)


if __name__ == '__main__':
   app.run(debug=True)