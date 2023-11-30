import csv
import tiktoken
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from tqdm.auto import tqdm
from uuid import uuid4
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# create the length function
def tiktoken_len(text):
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def create_index(csv_file_path):
    # Initialize an empty list to store provider data
    providers_data = []

   # Read the CSV file and store its contents in a list of dictionaries
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
    
        for row in reader:
            providers_data.append(dict(row))
    
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')

    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(providers_data[6]['about'])[:3]

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
    ]
    
    res = embed.embed_documents(texts)
    len(res), len(res[0])
    # find API key in console at app.pinecone.io
    YOUR_API_KEY = '7a2eff77-fb6e-4c40-aa04-d31f9d32e93c'
    # find ENV (cloud region) next to API key in console
    YOUR_ENV = 'gcp-starter'
    
    # index_name = "langchain-retrieval-augmentation"
    pinecone.init(
        api_key=YOUR_API_KEY,
        environment=YOUR_ENV
        )
    
    # Create a new Pinecone index
    index_name = 'langchain-augmentation'
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name) # Delete the existing index if it exists
    pinecone.create_index(
        name=index_name,
        metric='cosine',
        dimension=len(res[0]) # Replace with the actual dimension of your embeddings
    )
    index = pinecone.GRPCIndex(index_name)
    
    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(providers_data)):
        # first get metadata fields for this record
        metadata = {
            'id':record['id'],
            'name': record['name'],
            'phone':record['phone'],
            'headline': record['headline'],
            'isApproved':record['isApproved'],            
            'country': record['country'],
            'city': record['city'],
            'state': record['state'],
            'zip': record['zipCode'],
            'addressone': record['addressLineone'],
            'addresstwo': record['addressLinetwo'],
            'services_id': record['services_id'],
            'services_name': record['services_name'],
            'services_desc': record['services_description'],
            'School':record['schoolName'],
            'programme':record['programme'],
            'duration':record['education_duration'],
            'education_description':record['education_description'],
            'company':record['companyName'],
            'position':record['position'],
            'experience':record['workExperience_duration'],
            'jd':record['job_description'],
            'achievements_description':record['achievements_description'],

        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['about'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{
            "chunk": j, "about": text, **metadata
        } for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    return "Index creation completed."