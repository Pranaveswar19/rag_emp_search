from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from supabase import create_client
from groq import Groq
import config
import json
from datetime import datetime, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load model to reduce startup memory
_model = None
def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model

# Lazy load clients to reduce startup memory
_supabase = None
_groq_client = None

def get_supabase():
    global _supabase
    if _supabase is None:
        _supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client

class SearchRequest(BaseModel):
    query: str

def extract_filters(query):
    try:
        groq_client = get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """Extract filters from employee search queries. Return ONLY a JSON object, nothing else.

Available fields:
- skills: array of specific programming languages, frameworks, or technologies (e.g., Python, React, AWS, Docker)
- join_date_after: date string in YYYY-MM-DD format
- department: string (Engineering, Data, Design, Product, Marketing, Sales, HR, Finance, Security, QA, Operations, Legal, Business Development, Customer Success)
- min_experience: integer representing years

IMPORTANT: "skills" should contain ONLY technology names. Ignore words like: developers, engineers, Backend, Frontend, Full-stack, Senior, Junior, people, staff.

Today is """ + datetime.now().strftime("%Y-%m-%d") + """ for date calculations."""
                },
                {
                    "role": "user",
                    "content": "Python developers"
                },
                {
                    "role": "assistant",
                    "content": '{"skills": ["Python"]}'
                },
                {
                    "role": "user",
                    "content": "Backend developers in Engineering department"
                },
                {
                    "role": "assistant",
                    "content": '{"department": "Engineering"}'
                },
                {
                    "role": "user",
                    "content": "React developers with 5+ years experience"
                },
                {
                    "role": "assistant",
                    "content": '{"skills": ["React"], "min_experience": 5}'
                },
                {"role": "user", "content": query}
            ],
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        filters = json.loads(content)
        
        # Post-process: remove non-technology words from skills
        if "skills" in filters and filters["skills"]:
            blacklist = {
                "developers", "developer", "engineers", "engineer", "backend", "frontend",
                "full-stack", "fullstack", "senior", "junior", "staff", "people", "expert",
                "specialist", "lead", "principal", "architect", "manager", "programmers",
                "programmer", "coders", "coder"
            }
            filters["skills"] = [
                skill for skill in filters["skills"]
                if skill.lower() not in blacklist
            ]
        
        return filters
    except Exception as e:
        print(f"Filter extraction error: {e}")
        return {}

def search_employees(query_embedding, filters, limit=20):
    supabase = get_supabase()
    query_builder = supabase.table("employees").select("*, employee_embeddings(embedding)")
    
    # Apply hard filters for date, department, and experience
    if "join_date_after" in filters:
        query_builder = query_builder.gte("join_date", filters["join_date_after"])
    
    if "department" in filters:
        query_builder = query_builder.eq("department", filters["department"])
    
    if "min_experience" in filters:
        query_builder = query_builder.gte("experience_years", filters["min_experience"])
    
    result = query_builder.execute()
    employees_with_embeddings = result.data
    
    # Apply soft skill filter: if skills specified, employee must have at least one matching skill
    if "skills" in filters and filters["skills"]:
        employees_with_embeddings = [
            emp for emp in employees_with_embeddings
            if any(skill.lower() in [s.lower() for s in emp.get("skills", [])] for skill in filters["skills"])
        ]
    
    scored_employees = []
    for emp in employees_with_embeddings:
        emp_embedding = None
        
        if "employee_embeddings" in emp:
            if isinstance(emp["employee_embeddings"], list) and len(emp["employee_embeddings"]) > 0:
                emb_data = emp["employee_embeddings"][0].get("embedding")
            elif isinstance(emp["employee_embeddings"], dict):
                emb_data = emp["employee_embeddings"].get("embedding")
            else:
                continue
            
            if isinstance(emb_data, str):
                try:
                    emp_embedding = json.loads(emb_data)
                except json.JSONDecodeError:
                    continue
            elif isinstance(emb_data, list):
                emp_embedding = emb_data
        
        if emp_embedding and isinstance(emp_embedding, list):
            similarity = sum(float(a) * float(b) for a, b in zip(query_embedding, emp_embedding))
            scored_employees.append((similarity, emp))
    
    scored_employees.sort(reverse=True, key=lambda x: x[0])
    return [emp for _, emp in scored_employees[:limit]]

def generate_summary(query, employees):
    if not employees:
        return "No employees found matching your criteria."
    
    emp_text = "\n".join([
        f"- {emp['name']}: {', '.join(emp['skills'][:3])} | {emp['department']} | Joined: {emp['join_date']}"
        for emp in employees[:10]
    ])
    
    try:
        groq_client = get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Summarize employee search results in 1-2 sentences."},
                {"role": "user", "content": f"Query: '{query}'\n\nResults:\n{emp_text}"}
            ],
            model=config.LLM_MODEL,
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content
    except:
        return f"Found {len(employees)} employees matching your search."

@app.post("/api/search")
async def search(request: SearchRequest):
    filters = extract_filters(request.query)
    model = get_model()
    query_embedding = model.encode(request.query).tolist()
    employees = search_employees(query_embedding, filters)
    summary = generate_summary(request.query, employees)
    
    clean_employees = []
    for emp in employees:
        clean_emp = {
            "id": emp["id"],
            "name": emp["name"],
            "skills": emp["skills"],
            "department": emp["department"],
            "join_date": emp["join_date"],
            "experience_years": emp["experience_years"],
            "bio": emp["bio"],
            "email": emp["email"]
        }
        clean_employees.append(clean_emp)
    
    return {
        "success": True,
        "employees": clean_employees,
        "summary": summary,
        "count": len(clean_employees),
        "filters_applied": filters
    }

@app.get("/")
async def root():
    return {"status": "running", "service": "Employee RAG Search API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
