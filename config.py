import os
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Validate required environment variables
if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL environment variable is required")
if not SUPABASE_KEY:
    raise ValueError("SUPABASE_KEY environment variable is required")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

# Groq LLM (fast, free)
LLM_MODEL = "llama-3.1-8b-instant"
# Hash-based embedding dimension (lightweight, free)
EMBEDDING_DIMENSION = 384
TABLE_NAME = "employees"
EMBEDDINGS_TABLE = "employee_embeddings"
