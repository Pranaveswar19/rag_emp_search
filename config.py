import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TABLE_NAME = "employees"
EMBEDDINGS_TABLE = "employee_embeddings"
