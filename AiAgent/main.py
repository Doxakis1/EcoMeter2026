import os
import getpass
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import json
import asyncio

# Import watsonx stuff
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai import Credentials

# Import langchain stuff
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import nltk
nltk.download('averaged_perceptron_tagger_eng')

# Import gemini stuff
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env file
try:
    env_file = os.environ["DOTENV_FILE"]
except:
	env_file = ".env"
load_dotenv(env_file)

# Watsonx embeddings model initialization
watsonx_api_key = os.environ["WATSONX_API_KEY"]
watsonx_project_id = os.environ["WATSONX_PROJECT_ID"]
watsonx_url = os.environ["WATSONX_URL"]
watsonx_embedding_model_id="ibm/granite-embedding-278m-multilingual"
embeddings = WatsonxEmbeddings(
    model_id=watsonx_embedding_model_id,
    url=watsonx_url,
    apikey=watsonx_api_key,
    project_id=watsonx_project_id,
)

# Gemini model initialization
google_api_key = os.environ["GEMINI_API_KEY"]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=google_api_key,
    temperature=0.7,
)

def load_single_json_file_as_documents(file_path):
    all_documents = []
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"❌ Could not find the data file at: {path.absolute()}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Case 1: The file contains a list of multiple chunks/entries (Most Likely)
        if isinstance(data, list):
            print(f"📁 Found a list containing {len(data)} entries inside {path.name}...")
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # Fallback chain to find text contents dynamically
                    content = item.get("text") or item.get("page_content") or item.get("content") or str(item)
                    # Extract metadata if it exists, or use the item itself as metadata fields
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else item.copy()
                else:
                    content = str(item)
                    metadata = {}

                # Clean up whitespace layout anomalies
                cleaned_content = " ".join(content.split())

                # Stamp tracking parameters to metadata
                metadata["id"] = idx
                metadata["source_file"] = path.name

                # Append a dedicated separate document object
                all_documents.append(Document(page_content=cleaned_content, metadata=metadata))

        # Case 2: The file is a single big dictionary with entries mapped under a specific key
        elif isinstance(data, dict):
            print(f"📁 Found a single dictionary root inside {path.name}. Searching for entry groups...")
            # Look for common array keys like 'entries', 'chunks', 'segments', 'documents'
            possible_keys = ["entries", "chunks", "segments", "documents", "data"]
            target_list = None

            for key in possible_keys:
                if key in data and isinstance(data[key], list):
                    target_list = data[key]
                    break

            if target_list:
                for idx, item in enumerate(target_list):
                    content = item.get("text") or item.get("page_content") or item.get("content") or str(item)
                    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else item.copy()
                    cleaned_content = " ".join(content.split())
                    metadata["id"] = idx
                    metadata["source_file"] = path.name
                    all_documents.append(Document(page_content=cleaned_content, metadata=metadata))
            else:
                # Fallback: Treat the whole dictionary as one single root document entry
                content = data.get("text") or data.get("page_content") or str(data)
                metadata = data.copy()
                metadata["id"] = 0
                metadata["source_file"] = path.name
                all_documents.append(Document(page_content=" ".join(content.split()), metadata=metadata))

        print(f"✅ Cleanly separated data into {len(all_documents)} initial LangChain Documents.")
        return all_documents

    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Failed to parse JSON file: {e}")
        return []

# 1. Run the single file extractor pointing directly to your JSON
# Update the path string if it's located somewhere else relative to main.py
documents = load_single_json_file_as_documents("../AiChunkedData/all_segments_ready.json")

# 2. Split with safe token sizing to stay clear of the 512 IBM Watsonx limits
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,  # Keeping character sizes tight (~225 tokens) to protect IBM limits
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# 3. Streamlined embedding ingestion
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
print(f"🚀 Database updated! Your single JSON file successfully generated {len(docs)} searchable vector entries.")

old_template = """
You are a lawyer reviewing documents/or ideas for new businesses. You specialize in EU law, and specifically in \
biodiversity. Your job is to analyze the user input, and give valuable output about whethere there is any gaps, or \
biodiversity risks or cornerns.
Answer style should be proffesional. Give citations and be thorough, do not hallucinate. Give the user questions to \
provoke improvment and make the business more compliant with European biodiversity laws.
Ideal Answer Length 10-15 sentences.\n\n{context}\nQuestion: {question}\nAnswer:
"""
template = """
You are a law expert on EU contract law and biodiversity.
You will be given business ideas and your job is to find possible places where the business idea could be breaking biodiversity rules. Ellaborate and give the user questions to clarify so that their business can be fully legal without any issues.
Answer questions with deep knowledge of the law, cases and give detailed citations.
Give your answers back in a markdown format that I can use in my html site.
Ideal Answer Length 10-15 sentences.\n\n{context}\nQuestion: {question}\nAnswer:
"""
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


prompt = ChatPromptTemplate.from_template(template)
chain = (
	{"context": retriever | format_docs, "question": RunnablePassthrough()}
	| prompt
    | llm
    | StrOutputParser()
)
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs


class RAGServerHandler(BaseHTTPRequestHandler):

    # 1. Handle the GET "/" Route
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()

            response = {"status": "online", "message": "Καλώς ήρθες στο native API σου!"}
            self.wfile.write(json.dumps(response, ensure_ascii=False).encode("utf-8"))

        # 2. Handle the GET "/prompt2?user_input=..." Route
        elif parsed_url.path == "/prompt":
            query_params = parse_qs(parsed_url.query)
            user_input_list = query_params.get("user_input", None)

            if not user_input_list:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Missing 'user_input' query parameter.")
                return

            user_input = user_input_list[0]

            # Run your LangChain query pipeline
            print(f"Running invoke with input: {user_input}")
            answer = chain.invoke(user_input)
            print(answer)

            response_data = {
                "status": "success",
                "user_prompt": user_input,
                "server_response": answer
            }

            # Send HTTP Headers
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()

            # Output formatted response to terminal and browser
            print(">>> [SERVER RESPONSE] Ο server responds with JSON:")
            print(json.dumps(response_data, indent=4, ensure_ascii=False))
            self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode("utf-8"))

        else:
            # Handle 404 Not Found
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Route Not Found")

# 3. Boot the native HTTP Server
if __name__ == "__main__":
    host_name = "0.0.0.0"
    port = 9090

    server = HTTPServer((host_name, port), RAGServerHandler)
    print(f"🚀 Server running cleanly without Uvicorn at http://{host_name}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()
