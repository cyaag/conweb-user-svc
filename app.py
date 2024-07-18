import boto3
import os
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.chains.question_answering import load_qa_chain
from langchain_pinecone import PineconeVectorStore
import pinecone
from pinecone import Pinecone, ServerlessSpec
from flask import Flask, jsonify, request

# app = Flask(__name__)

# Set the cache directory to /tmp
os.environ['TRANSFORMERS_CACHE'] = '/tmp'

bedrock_runtime = boto3.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1"
)

# Instantiate a bedrock instance
bedrock = boto3.client(
    service_name = "bedrock",
    region_name = "us-east-1"
)

# modelId = ['anthropic.claude-v2:1','amazon.titan-text-premier-v1:0', 'cohere.command-r-plus-v1:0']
modelId = str(os.environ.get("MODEL_ID", default=None))
accept = 'application/json'
contentType = 'application/json'

bedrock_embeddings = BedrockEmbeddings(client=bedrock_runtime)

# Initialize Pinecone
api_key = str(os.environ.get("PINE_API", default=None))
pc = Pinecone(api_key=api_key)
# pinecone.init(api_key=api_key, environment='ap-south-1')
index_name = str(os.environ.get("PINE_INDEX", default=None))
# vectorstore = PineconeVectorStore(index_name=index_name, embedding=bedrock_embeddings)
index = pc.Index(index_name)

# Initialize LLM
llm = Bedrock(
    model_id=modelId,
    client=bedrock_runtime
)

os.environ['PINECONE_API_KEY'] = api_key

# @app.route("/get/data/<query>", methods=['GET'])
def fetch_data_load_chain(query):
    try:
        # Embed the query
        query_embedding = bedrock_embeddings.embed_query(query)

        # Query Pinecone for similar vectors
        print('extracting embeding')
        # response = index.query(queries=[query_embedding], top_k=10, include_metadata=True, include_values=True)
        # print("response = ", response)
        vectorsearch = PineconeVectorStore(
            # query,
            embedding=bedrock_embeddings,
            index_name = index_name,
        )
        # # Extract relevant information from the response
        # results = [match['metadata']['text'] for match in response['matches']]
        # context = " ".join(results)
        context = vectorsearch.similarity_search(query)
        # context = context[0]
        # Load the QA chain
        print(context)
        chain = load_qa_chain(llm, chain_type="stuff")
        # chaint = load_qa_chain(llmt, chain_type="stuff")
        # Generate the answer
        answer = chain.run(input_documents=context, question=query)
        # answer = chaint.run(input_documents=context, question=query)

        return jsonify({"message": answer}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    
def lambda_handler(event, context):
    print(event)
    # logging.info("Event: %s", json.dumps(event))
    # body = json.loads(event['body'])

    # Call create_user directly with the body
    return fetch_data_load_chain(event)
# if __name__ == '__main__':
#     app.run(debug=True)