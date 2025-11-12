from supabase import create_client
import config
from fake_employees import FAKE_EMPLOYEES
import numpy as np

supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

def create_simple_embedding(text, dimension=None):
    """Lightweight hash-based embedding - 100% free"""
    if dimension is None:
        dimension = config.EMBEDDING_DIMENSION
    words = text.lower().split()
    embedding = np.zeros(dimension)
    for i, word in enumerate(words):
        hash_val = hash(word) % dimension
        embedding[hash_val] += 1.0 / (i + 1)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()

def create_employee_text(emp):
    return f"""
    Name: {emp['name']}
    Skills: {', '.join(emp['skills'])}
    Department: {emp['department']}
    Experience: {emp['experience_years']} years
    Bio: {emp['bio']}
    Joined: {emp['join_date']}
    """

def embed_employees():
    print("Deleting existing data...")
    supabase.table("employee_embeddings").delete().neq("id", 0).execute()
    supabase.table("employees").delete().neq("id", 0).execute()
    
    print(f"Embedding {len(FAKE_EMPLOYEES)} employees with FREE hash-based embeddings...")
    
    for emp in FAKE_EMPLOYEES:
        supabase.table("employees").insert({
            "id": emp["id"],
            "name": emp["name"],
            "skills": emp["skills"],
            "department": emp["department"],
            "join_date": emp["join_date"],
            "experience_years": emp["experience_years"],
            "bio": emp["bio"],
            "email": emp["email"]
        }).execute()
        
        text = create_employee_text(emp)
        embedding = create_simple_embedding(text)
        
        supabase.table("employee_embeddings").insert({
            "id": emp["id"],
            "embedding": embedding
        }).execute()
        
        print(f"✓ Embedded: {emp['name']}")
    
    print(f"\nSuccessfully embedded {len(FAKE_EMPLOYEES)} employees!")

if __name__ == "__main__":
    embed_employees()
