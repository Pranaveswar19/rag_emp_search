from sentence_transformers import SentenceTransformer
from supabase import create_client
import config
from fake_employees import FAKE_EMPLOYEES

model = SentenceTransformer(config.EMBEDDING_MODEL)
supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)

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
    
    print(f"Embedding {len(FAKE_EMPLOYEES)} employees...")
    
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
        embedding = model.encode(text).tolist()
        
        supabase.table("employee_embeddings").insert({
            "id": emp["id"],
            "embedding": embedding
        }).execute()
        
        print(f"✓ Embedded: {emp['name']}")
    
    print(f"\nSuccessfully embedded {len(FAKE_EMPLOYEES)} employees!")

if __name__ == "__main__":
    embed_employees()
