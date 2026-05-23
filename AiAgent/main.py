import os
import getpass
from dotenv import load_dotenv

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
import asyncio
import fastapi
from fastapi import FastAPI
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

### THIS CHANGES TO OUR CHUNKED DATA
URLS_DICTIONARY = {
    "ibm.com_events_think_faq.html": "https://www.ibm.com/events/think/faq",
    "events_think_agenda.html": "https://www.ibm.com/events/think/agenda",
    "products_watsonx_ai.html": "https://www.ibm.com/products/watsonx-ai",
    "products_watsonx_ai_foundation_models.html": "https://www.ibm.com/products/watsonx-ai/foundation-models",
    "watsonx_pricing.html": "https://www.ibm.com/watsonx/pricing",
    "watsonx.html": "https://www.ibm.com/watsonx",
    "products_watsonx_data.html": "https://www.ibm.com/products/watsonx-data",
    "products_watsonx_assistant.html": "https://www.ibm.com/products/watsonx-assistant",
    "products_watsonx_code_assistant.html": "https://www.ibm.com/products/watsonx-code-assistant",
    "products_watsonx_orchestrate.html": "https://www.ibm.com/products/watsonx-orchestrate",
    "products_watsonx_governance.html": "https://www.ibm.com/products/watsonx-governance",
    "granite_code_models_open_source.html": "https://research.ibm.com/blog/granite-code-models-open-source",
    "red_hat_enterprise_linux_ai.html": "https://www.redhat.com/en/about/press-releases/red-hat-delivers-accessible-open-source-generative-ai-innovation-red-hat-enterprise-linux-ai",
    "model_choice.html": "https://www.ibm.com/blog/announcement/enterprise-grade-model-choices/",
    "democratizing.html": "https://www.ibm.com/blog/announcement/democratizing-large-language-model-development-with-instructlab-support-in-watsonx-ai/",
    "ibm_consulting_expands_ai.html": "https://newsroom.ibm.com/Blog-IBM-Consulting-Expands-Capabilities-to-Help-Enterprises-Scale-AI",
    "ibm_data_product_hub.html": "https://www.ibm.com/products/data-product-hub",
    "ibm_price_performance_data.html": "https://www.ibm.com/blog/announcement/delivering-superior-price-performance-and-enhanced-data-management-for-ai-with-ibm-watsonx-data/",
    "ibm_bi_adoption.html": "https://www.ibm.com/blog/a-new-era-in-bi-overcoming-low-adoption-to-make-smart-decisions-accessible-for-all/",
    "watsonx_code_assistant_for_z.html": "https://www.ibm.com/blog/announcement/ibm-watsonx-code-assistant-for-z-accelerate-the-application-lifecycle-with-generative-ai-and-automation/",
    "code_assistant_for_java.html": "https://www.ibm.com/blog/announcement/watsonx-code-assistant-java/",
    "code_assistant_for_orchestrate.html": "https://www.ibm.com/blog/announcement/watsonx-orchestrate-ai-z-assistant/",
    "accelerating_gen_ai.html": "https://newsroom.ibm.com/Blog-How-IBM-Cloud-is-Accelerating-Business-Outcomes-with-Gen-AI",
    "watsonx_open_source.html": "https://newsroom.ibm.com/2024-05-21-IBM-Unveils-Next-Chapter-of-watsonx-with-Open-Source,-Product-Ecosystem-Innovations-to-Drive-Enterprise-AI-at-Scale",
    "ibm_concert.html": "https://www.ibm.com/products/concert",
    "ibm_consulting_advantage_news.html": "https://newsroom.ibm.com/2024-01-17-IBM-Introduces-IBM-Consulting-Advantage,-an-AI-Services-Platform-and-Library-of-Assistants-to-Empower-Consultants",
    "ibm_consulting_advantage_info.html": "https://www.ibm.com/consulting/info/ibm-consulting-advantage"
}
COLLECTION_NAME = "askibm_think_2024"
documents = []

for url in list(URLS_DICTIONARY.values()):
    loader = UnstructuredURLLoader(urls=[url])
    data = loader.load()
    documents += data

doc_id = 0
for doc in documents:
    doc.page_content = " ".join(doc.page_content.split()) # remove white space

    doc.metadata["id"] = doc_id #make a document id and add it to the document metadata

	#print(doc.metadata)
    doc_id += 1

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever()
template = """Generate a summary of the context that answers the question. Explain the answer in multiple steps if possible.
Answer style should match the context. Ideal Answer Length 2-3 sentences.\n\n{context}\nQuestion: {question}\nAnswer:
"""
prompt = ChatPromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("GEMINI ANSWER")
print(chain.invoke("What is IBM Concert?"))

