from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from supabase import create_client
from groq import Groq
from openai import OpenAI
import config
import json
from datetime import datetime, timedelta
from pathlib import Path

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load clients
_supabase = None
_groq_client = None
_openai_client = None

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

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _openai_client

def create_openai_embedding(text):
    """Create high-quality embedding using OpenAI API"""
    client = get_openai_client()
    response = client.embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

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
- join_date: exact date string in YYYY-MM-DD format (use this when query says "joined on" or "hired on" specific date)
- join_date_after: date string in YYYY-MM-DD format (use this when query says "joined after" or "since")
- join_date_before: date string in YYYY-MM-DD format (use this when query says "joined before" or "prior to")
- department: string (Engineering, Data, Design, Product, Marketing, Sales, HR, Finance, Security, QA, Operations, Legal, Business Development, Customer Success)
- min_experience: integer representing years (use for "at least X years" or "X+ years")
- sort_by: string - either "experience_desc" (most experienced), "experience_asc" (least experienced), or "join_date_desc" (most recent)

IMPORTANT: 
- "skills" should contain ONLY technology names. Ignore words like: developers, engineers, Backend, Frontend, Full-stack, Senior, Junior, people, staff.
- If query says "joined on [date]" use "join_date", NOT "join_date_after"
- If query says "joined after [date]" or "since [date]" use "join_date_after"
- If query says "joined before [date]" or "prior to [date]" use "join_date_before"
- If query says "most experienced" or "most years" use "sort_by": "experience_desc" (DO NOT use min_experience)
- If query says "least experienced" or "newest" use "sort_by": "experience_asc"
- If query says "most recent" or "latest hires" use "sort_by": "join_date_desc"

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
                {
                    "role": "user",
                    "content": "Employees who joined on 2023-09-15"
                },
                {
                    "role": "assistant",
                    "content": '{"join_date": "2023-09-15"}'
                },
                {
                    "role": "user",
                    "content": "Employees who joined before 2024"
                },
                {
                    "role": "assistant",
                    "content": '{"join_date_before": "2024-01-01"}'
                },
                {
                    "role": "user",
                    "content": "Person with most years of experience"
                },
                {
                    "role": "assistant",
                    "content": '{"sort_by": "experience_desc"}'
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

def search_employees(query_embedding, filters, limit=50):
    supabase = get_supabase()
    query_builder = supabase.table("employees").select("*, employee_embeddings(embedding)")
    
    # Apply hard filters for date, department, and experience
    if "join_date" in filters:
        # Exact date match
        query_builder = query_builder.eq("join_date", filters["join_date"])
    elif "join_date_after" in filters:
        # Date range (after)
        query_builder = query_builder.gte("join_date", filters["join_date_after"])
    elif "join_date_before" in filters:
        # Date range (before)
        query_builder = query_builder.lt("join_date", filters["join_date_before"])
    
    if "department" in filters:
        query_builder = query_builder.eq("department", filters["department"])
    
    if "min_experience" in filters:
        query_builder = query_builder.gte("experience_years", filters["min_experience"])
    
    result = query_builder.execute()
    employees_with_embeddings = result.data
    
    # Apply skill filter MORE STRICTLY
    # If skills are specified, REQUIRE at least one exact match
    if "skills" in filters and filters["skills"]:
        # Exact skill matching - employee must have at least one of the requested skills
        employees_with_embeddings = [
            emp for emp in employees_with_embeddings
            if any(skill.lower() in [s.lower() for s in emp.get("skills", [])] for skill in filters["skills"])
        ]
        
        # If no exact matches found, return empty (don't fallback to semantic)
        if not employees_with_embeddings:
            return []
    
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
    
    # Apply sorting if specified
    if "sort_by" in filters:
        if filters["sort_by"] == "experience_desc":
            # Sort by experience (most to least)
            scored_employees.sort(key=lambda x: x[1]["experience_years"], reverse=True)
        elif filters["sort_by"] == "experience_asc":
            # Sort by experience (least to most)
            scored_employees.sort(key=lambda x: x[1]["experience_years"])
        elif filters["sort_by"] == "join_date_desc":
            # Sort by join date (most recent first)
            scored_employees.sort(key=lambda x: x[1]["join_date"], reverse=True)
    else:
        # Default: sort by similarity score
        scored_employees.sort(reverse=True, key=lambda x: x[0])
    
    return [emp for _, emp in scored_employees[:limit]]

def generate_summary(query, employees):
    if not employees:
        return "No employees found matching your criteria."
    
    # Limit to top 5 for summary to avoid hallucinations
    top_employees = employees[:5]
    
    emp_text = "\n".join([
        f"- {emp['name']}: {', '.join(emp['skills'][:4])} | {emp['department']} | {emp['experience_years']} years exp | Joined: {emp['join_date']}"
        for emp in top_employees
    ])
    
    try:
        groq_client = get_groq_client()
        response = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": """Summarize employee search results in 1-2 sentences. 
                    
IMPORTANT RULES:
- State the EXACT number of total results found
- Mention key skills or departments from the TOP results only
- DO NOT make up names, dates, or skills
- DO NOT mention dates beyond 2024
- Keep it factual and brief"""
                },
                {
                    "role": "user", 
                    "content": f"Query: '{query}'\n\nTotal results: {len(employees)}\n\nTop 5 employees:\n{emp_text}"
                }
            ],
            model=config.LLM_MODEL,
            temperature=0,
            max_tokens=100
        )
        return response.choices[0].message.content
    except:
        return f"Found {len(employees)} employees matching '{query}'."

@app.post("/api/search")
async def search(request: SearchRequest):
    filters = extract_filters(request.query)
    # Use OpenAI for high-quality embeddings
    query_embedding = create_openai_embedding(request.query)
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
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}

class EmployeeCreate(BaseModel):
    name: str
    skills: list[str]
    department: str
    join_date: str
    experience_years: int
    bio: str
    email: str

def create_employee_embedding(employee_data):
    """Helper function to create high-quality OpenAI embedding"""
    text = f"""
    Name: {employee_data['name']}
    Skills: {', '.join(employee_data['skills'])}
    Department: {employee_data['department']}
    Experience: {employee_data['experience_years']} years
    Bio: {employee_data['bio']}
    Joined: {employee_data['join_date']}
    """
    return create_openai_embedding(text)

@app.post("/api/admin/add_employee")
async def add_employee(employee: EmployeeCreate):
    """Add a new employee with automatic embedding generation"""
    try:
        supabase = get_supabase()
        
        # Get next available ID
        result = supabase.table("employees").select("id").order("id", desc=True).limit(1).execute()
        next_id = (result.data[0]["id"] + 1) if result.data else 1
        
        # Insert employee
        emp_data = {
            "id": next_id,
            "name": employee.name,
            "skills": employee.skills,
            "department": employee.department,
            "join_date": employee.join_date,
            "experience_years": employee.experience_years,
            "bio": employee.bio,
            "email": employee.email
        }
        supabase.table("employees").insert(emp_data).execute()
        
        # Create and insert embedding
        embedding = create_employee_embedding(emp_data)
        supabase.table("employee_embeddings").insert({
            "id": next_id,
            "embedding": embedding
        }).execute()
        
        return {
            "success": True,
            "message": f"Employee {employee.name} added successfully",
            "id": next_id
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.put("/api/admin/update_employee/{employee_id}")
async def update_employee(employee_id: int, employee: EmployeeCreate):
    """Update employee and automatically re-create embedding"""
    try:
        supabase = get_supabase()
        
        # Update employee data
        emp_data = {
            "name": employee.name,
            "skills": employee.skills,
            "department": employee.department,
            "join_date": employee.join_date,
            "experience_years": employee.experience_years,
            "bio": employee.bio,
            "email": employee.email
        }
        supabase.table("employees").update(emp_data).eq("id", employee_id).execute()
        
        # Re-create embedding
        emp_data["id"] = employee_id
        embedding = create_employee_embedding(emp_data)
        supabase.table("employee_embeddings").update({
            "embedding": embedding
        }).eq("id", employee_id).execute()
        
        return {
            "success": True,
            "message": f"Employee {employee.name} updated successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.delete("/api/admin/delete_employee/{employee_id}")
async def delete_employee(employee_id: int):
    """Delete employee and their embedding"""
    try:
        supabase = get_supabase()
        
        # Delete embedding first (foreign key constraint)
        supabase.table("employee_embeddings").delete().eq("id", employee_id).execute()
        
        # Delete employee
        supabase.table("employees").delete().eq("id", employee_id).execute()
        
        return {
            "success": True,
            "message": f"Employee {employee_id} deleted successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

class WebhookPayload(BaseModel):
    employee_id: int
    action: str

@app.post("/api/admin/webhook/re-embed")
async def webhook_re_embed(payload: WebhookPayload):
    """
    Webhook endpoint called by Supabase trigger when employee data changes.
    Automatically re-creates embeddings when HR updates Supabase directly.
    """
    try:
        supabase = get_supabase()
        
        # Get employee data
        result = supabase.table("employees").select("*").eq("id", payload.employee_id).execute()
        if not result.data:
            return {"success": False, "error": "Employee not found"}
        
        employee = result.data[0]
        
        # Create new embedding
        embedding = create_employee_embedding(employee)
        
        # Update or insert embedding
        supabase.table("employee_embeddings").upsert({
            "id": payload.employee_id,
            "embedding": embedding
        }).execute()
        
        return {
            "success": True,
            "message": f"Re-embedded employee {payload.employee_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/api/admin/re-embed-all")
@app.post("/api/admin/re-embed-all")
async def re_embed_all():
    """
    Re-embed ALL employees in the database.
    Use this after migration or to refresh all embeddings.
    """
    try:
        supabase = get_supabase()
        
        # Get all employees
        result = supabase.table("employees").select("*").execute()
        employees = result.data
        
        if not employees:
            return {"success": False, "error": "No employees found in database"}
        
        success_count = 0
        for employee in employees:
            try:
                # Create embedding
                embedding = create_employee_embedding(employee)
                
                # Upsert embedding
                supabase.table("employee_embeddings").upsert({
                    "id": employee["id"],
                    "embedding": embedding
                }).execute()
                
                success_count += 1
            except Exception as e:
                print(f"Error embedding employee {employee['id']}: {e}")
        
        return {
            "success": True,
            "message": f"Re-embedded {success_count} out of {len(employees)} employees",
            "total": len(employees),
            "success_count": success_count
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
