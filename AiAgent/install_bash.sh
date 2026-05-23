#/bin/env /bin/bash

sudo apt install python3.12-venv -y
python3 -m venv .venv
source .venv/bin/activate
pip install dotenv
pip install langchain-google-genai
pip install langchain
pip install langchain_chroma
pip install langchain-community
pip install -U langchain_ibm
pip install unstructured
pip install "ibm-watson-machine-learning>=1.0.327"
pip install nltk
pip install fastapi
pip install google-cloud-aiplatform
pip install google-generativeai
