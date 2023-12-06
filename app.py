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
    queries = request.args.getlist("query")

    # Fetch the 5 most suitable providers from your Pinecone index...
    # This will depend on your specific implementation
    results = fetch_providers(queries)

    # Extract the top 3 results.
    # top_results = results

    # Convert the results to JSON format.
    json_results = []
    for result in results:
        json_providers = []
        for provider in result['recomendation']:
          json_providers.append({   
       "id": provider["id"],
      "Name": provider["Name"],
      "phone": provider["phone"],
      "headline": provider["headline"],
      "isApproved": provider["isApproved"],
      "address": {
                "country": provider["address"]["country"],
                "city": provider["address"]["city"],
                "state": provider["address"]["state"],
                "zipCode": provider["address"]["zipCode"],
                "addressLineOne": provider["address"]["addressLineOne"],
                "addressLineTwo": provider["address"]["addressLineTwo"],
            },
      "services":{
        'services_id': provider["services"]['services_id'],
            'services_name': provider["services"]['services_name'],
            'services_desc': provider["services"]['services_desc'],
      },
      "education":{
        "School": provider["education"]["School"],
        "programme": provider["education"]["programme"],
        "duration": provider["education"]["duration"],
        'education_description':provider["education"]['education_description'],
      },
      "workExperience": {
        "companyName": provider["workExperience"]["companyName"],
        "position": provider["workExperience"]["position"],
        "experience": provider["workExperience"]["experience"],
        "JD": provider["workExperience"]["JD"],
      },
      "achievements_description": provider["achievements_description"],
      "about": provider["about"],
        })
        json_results.append({
            "id": result["id"],
            "query": result["query"],
            "active": result["active"],
            "recomendation": json_providers
        })

    return json.dumps(json_results)


if __name__ == '__main__':
   app.run(debug=True)