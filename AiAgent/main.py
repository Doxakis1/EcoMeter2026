import os
import getpass
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

def load_json_from_folder(folder_path):
    all_data = []

    # Convert string path to a Path object
    target_dir = Path(folder_path)

    # Loop through all files ending in .json inside the folder
    for file_path in target_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_data.append(data)
                print(f"✅ Successfully loaded: {file_path.name}")
        except (json.JSONDecodeError, IOError) as e:
            print(f"❌ Failed to read {file_path.name}: {e}")

    return all_data
documents = load_json_from_folder("../AiChunkedData")

doc_id = 0
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split()) # remove white space
    doc.metadata["id"] = doc_id #make a document id and add it to the document metadata
	#print(doc.metadata)
    doc_id += 1

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=10)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
template = """
You are a lawyer reviewing documents/or ideas for new businesses. You specialize in EU law, and specifically in \
biodiversity. Your job is to analyze the user input, and give valuable output about whethere there is any gaps, or \
biodiversity risks or cornerns.
Answer style should be proffesional. Give citations and be thorough, do not hallucinate. Give the user questions to \
provoke improvment and make the business more compliant with European biodiversity laws.
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

# ... [Keep all your URL scraping, Chroma setup, and 'chain' initialization here] ...

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
        elif parsed_url.path == "/prompt2":
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
    port = 8080

    server = HTTPServer((host_name, port), RAGServerHandler)
    print(f"🚀 Server running cleanly without Uvicorn at http://{host_name}:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()
