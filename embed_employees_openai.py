from openai import OpenAI
from supabase import create_client
import config
from fake_employees import FAKE_EMPLOYEES

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
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
    print("Checking existing data...")
    
    # Check if employees exist
    emp_count = supabase.table("employees").select("id", count="exact").execute()
    print(f"Found {emp_count.count} employees in database")
    
    if emp_count.count == 0:
        print("\nInserting employee data...")
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
            print(f"✓ Inserted: {emp['name']}")
    
    print(f"\nCreating OpenAI embeddings for {len(FAKE_EMPLOYEES)} employees...")
    
    for emp in FAKE_EMPLOYEES:
        text = create_employee_text(emp)
        
        # Create OpenAI embedding
        response = openai_client.embeddings.create(
            model=config.OPENAI_EMBEDDING_MODEL,
            input=text
        )
        embedding = response.data[0].embedding
        
        # Upsert embedding (insert or update)
        supabase.table("employee_embeddings").upsert({
            "id": emp["id"],
            "embedding": embedding
        }).execute()
        
        print(f"✓ Embedded: {emp['name']}")
    
    print(f"\n✅ Successfully embedded {len(FAKE_EMPLOYEES)} employees with OpenAI!")
    print(f"💰 Approximate cost: ${len(FAKE_EMPLOYEES) * 0.000002:.4f}")

if __name__ == "__main__":
    embed_employees()
