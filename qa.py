from flask import jsonify
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import openai
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone and OpenAI embeddings
YOUR_API_KEY = "d36342e7-2d57-4a19-be61-75d73dc52f98"
YOUR_ENV = "gcp-starter"
pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

# Load the Pinecone index
index_name = "langchain-retrieval-augmentation"
text_field = "description"
index = pinecone.Index(index_name)

# Create a vector store using the Pinecone index
vectorstore = Pinecone(index, embed.embed_query, text_field)

def fetch_providers(query):
  """Returns the complete details of providers from the pinecone index, up to 3 most relevant.

  Args:
    query: The user query.

  Returns:
    A list of dictionaries, each containing the complete details of a provider.
  """

  # Create a pinecone client.

  # Get the pinecone index for providers.
  index = pinecone.Index("langchain-retrieval-augmentation")

  # Search the index for providers that match the user query.
  results = vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
  )
# 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
  # Extract the provider details from the search results.
  providers = []
  for result in results:
    provider = {
      "id": result.metadata["id"],
      "Name": result.metadata["name"],
      "designation": result.metadata["designation"],
      "field": result.metadata["field"],
      "phone": result.metadata["phone"],
      "location": result.metadata["location"],
      "School": result.metadata["School"],
      "programme": result.metadata["programme"],
      "duration": result.metadata["duration"],
      "company": result.metadata["company"],
      "position": result.metadata["position"],
      "JD": result.metadata["jd"],
      "experience": result.metadata["experience"],
      "description": result.page_content,

    }
    providers.append(provider)
  print(providers)
  return providers

