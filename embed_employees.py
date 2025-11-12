from supabase import create_client
import numpy as np
import config
from fake_employees import FAKE_EMPLOYEES

supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

def create_simple_embedding(text, dimension=None):
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
    return f"""Name: {emp['name']}
    Skills: {', '.join(emp['skills'])}
    Department: {emp['department']}
    Experience: {emp['experience_years']} years
    Bio: {emp['bio']}
    Joined: {emp['join_date']}"""

def embed_employees():
    supabase.table("employee_embeddings").delete().neq("id", 0).execute()
    supabase.table("employees").delete().neq("id", 0).execute()
    
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
        
        embedding = create_simple_embedding(create_employee_text(emp))
        supabase.table("employee_embeddings").insert({
            "id": emp["id"],
            "embedding": embedding
        }).execute()
        print(f"✓ {emp['name']}")
    
    print(f"\nEmbedded {len(FAKE_EMPLOYEES)} employees")

if __name__ == "__main__":
    embed_employees()
