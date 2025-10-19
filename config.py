import os
from dotenv import load_dotenv

load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")
api_version = os.getenv("OPENAI_API_VERSION")
api_endpoint = os.getenv("OPENAI_AZURE_ENDPOINT")
api_model = os.getenv("OPENAI_AZURE_MODEL")
langsmith_key = os.getenv("LANGCHAIN_API_KEY")
 

