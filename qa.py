from flask import jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import openai
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize Pinecone and OpenAI embeddings
YOUR_API_KEY = "7a2eff77-fb6e-4c40-aa04-d31f9d32e93c"
YOUR_ENV = "gcp-starter"
pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# Load the Pinecone index
index_name = "langchain-augmentation"
text_field = "about"
index = pinecone.Index(index_name)

# Create a vector store using the Pinecone index
vectorstore = Pinecone(index, embed.embed_query, text_field)

def fetch_providers(queries):
  """Returns the complete details of providers from the pinecone index, up to 3 most relevant.

  Args:
    query: The user query.

  Returns:
    A list of dictionaries, each containing the complete details of a provider.
  """

  # Create a pinecone client.

  # Get the pinecone index for providers.
  # index = pinecone.Index("langchain-retrieval-augmentation")
  providers = []
  # Search the index for providers that match the user query.
  for query in queries:

        # Search the index for providers that match the user query.
        results = vectorstore.similarity_search(
            query,  # our search query
            k=3  # return 3 most relevant docs
        )
# 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
  # Extract the provider details from the search results.
        for result in results:
           provider = {
      "id": result.metadata["id"],
      "query":query,
      "Name": result.metadata["name"],
      "phone": result.metadata["phone"],
      "headline": result.metadata["headline"],
      "isApproved": result.metadata["isApproved"],
      "address": {
                "country": result.metadata["country"],
                "city": result.metadata["city"],
                "state": result.metadata["state"],
                "zipCode": result.metadata["zip"],
                "addressLineOne": result.metadata["addressone"],
                "addressLineTwo": result.metadata["addresstwo"],
            },
      "services":{
        'services_id': result.metadata['services_id'],
            'services_name': result.metadata['services_name'],
            'services_desc': result.metadata['services_desc'],
      },
      "education":{
        "School": result.metadata["School"],
        "programme": result.metadata["programme"],
        "duration": result.metadata["duration"],
        'education_description':result.metadata['education_description'],
      },
      "workExperience": {
        "companyName": result.metadata["company"],
        "position": result.metadata["position"],
        "experience": result.metadata["experience"],
        "JD": result.metadata["jd"],
      },
      "achievements_description": result.metadata["achievements_description"],
      "about": result.page_content,
    }
           providers.append(provider)
  print(providers)

  
  
  return providers

