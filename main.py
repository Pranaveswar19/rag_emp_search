from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from supabase import create_client
from groq import Groq
import config
import json
import numpy as np
from datetime import datetime
import time

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class SearchRequest(BaseModel):
    query: str

SEARCH_FUNCTION_SCHEMA = {
    "name": "search_employees",
    "description": """Search for employees based on skills, experience, department, and join date filters.

CRITICAL RULES:
1. When query says "X years experience" or "with X years" WITHOUT comparison words (like 'at least', '+', 'or more') → use operator '=' (exactly X years)
2. Only use '>=' or '<=' when query EXPLICITLY contains: 'at least', '+', 'or more', 'or less', 'minimum', 'maximum'
3. Each filter is independent - parse them separately

Examples:
- "5 years experience" → experience: {operator: "=", value: 5}
- "5+ years" → experience: {operator: ">=", value: 5}
- "devs with 5 years who joined before 2024" → experience: {operator: "=", value: 5}, join_date: {operator: "<", date: "2024"}
- "Python devs with at least 3 years" → skills: ["Python"], experience: {operator: ">=", value: 3}
""",
    "parameters": {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Programming languages, frameworks, or technologies (e.g., Python, React, AWS). Exclude role words like 'developer', 'engineer', 'devs'."
            },
            "department": {
                "type": "string",
                "enum": ["Engineering", "Data", "Design", "Product", "Marketing", "Sales", "HR", "Finance", "Security", "QA", "Operations", "Legal", "Business Development", "Customer Success"],
                "description": "Department name"
            },
            "experience": {
                "type": "object",
                "description": "Years of experience filter",
                "properties": {
                    "operator": {
                        "type": "string",
                        "enum": ["=", ">=", "<=", ">", "<", "between"],
                        "description": "IMPORTANT: Default to '=' unless query has explicit comparison words. '=' for 'X years'/'with X years'. '>=' for 'X+'/'at least X'/'or more'. '<=' for 'or less'/'at most'. '>' for 'above'/'more than'. '<' for 'below'/'less than'."
                    },
                    "value": {
                        "type": "integer",
                        "description": "Experience value in years"
                    },
                    "value2": {
                        "type": "integer",
                        "description": "Only for 'between' operator - the upper bound"
                    }
                },
                "required": ["operator", "value"]
            },
            "join_date": {
                "type": "object",
                "description": "Join date filter",
                "properties": {
                    "operator": {
                        "type": "string",
                        "enum": ["=", ">", "<", "between"],
                        "description": "Comparison operator: = (on specific date), > (after), < (before), between (date range)"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format or YYYY for year"
                    },
                    "date2": {
                        "type": "string",
                        "description": "Second date for 'between' operator in YYYY-MM-DD format"
                    }
                },
                "required": ["operator", "date"]
            },
            "sort_by": {
                "type": "string",
                "enum": ["experience_desc", "experience_asc", "join_date_desc"],
                "description": "Sort order: experience_desc (most experienced), experience_asc (least experienced), join_date_desc (most recent)"
            }
        }
    }
}

SKILL_BLACKLIST = {
    "developers", "developer", "engineers", "engineer", "backend", "frontend",
    "full-stack", "fullstack", "senior", "junior", "staff", "people", "expert",
    "specialist", "lead", "principal", "architect", "manager", "programmers",
    "programmer", "coders", "coder", "devs"
}

def normalize_date(date_str):
    if len(date_str) == 4:
        return f"{date_str}-01-01"
    return date_str

def convert_function_call_to_filters(func_args):
    filters = {}
    
    if "skills" in func_args and func_args["skills"]:
        skills = [s for s in func_args["skills"] if s.lower() not in SKILL_BLACKLIST]
        if skills:
            filters["skills"] = skills
    
    if "department" in func_args:
        filters["department"] = func_args["department"]
    
    if "experience" in func_args:
        exp = func_args["experience"]
        operator = exp["operator"]
        value = exp["value"]
        
        if operator == "=":
            filters["exact_experience"] = value
        elif operator == ">=":
            filters["min_experience"] = value
        elif operator == "<=":
            filters["max_experience"] = value
        elif operator == ">":
            filters["min_experience"] = value + 1
        elif operator == "<":
            filters["max_experience"] = value - 1
        elif operator == "between" and "value2" in exp:
            filters["min_experience"] = value
            filters["max_experience"] = exp["value2"]
    
    if "join_date" in func_args:
        jd = func_args["join_date"]
        operator = jd["operator"]
        date = normalize_date(jd["date"])
        
        if operator == "=":
            filters["join_date"] = date
        elif operator == ">":
            if len(jd["date"]) == 4:
                filters["join_date_after"] = f"{jd['date']}-12-31"
            else:
                filters["join_date_after"] = date
        elif operator == "<":
            if len(jd["date"]) == 4:
                filters["join_date_before"] = f"{jd['date']}-01-01"
            else:
                filters["join_date_before"] = date
        elif operator == "between" and "date2" in jd:
            filters["join_date_after"] = date
            filters["join_date_before"] = normalize_date(jd["date2"])
    
    if "sort_by" in func_args:
        filters["sort_by"] = func_args["sort_by"]
    
    return filters

def extract_filters(query, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = get_groq_client().chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts search filters from natural language queries."},
                    {"role": "user", "content": query}
                ],
                tools=[{"type": "function", "function": SEARCH_FUNCTION_SCHEMA}],
                tool_choice={"type": "function", "function": {"name": "search_employees"}},
                temperature=0
            )
            
            if response.choices[0].message.tool_calls:
                func_args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
                filters = convert_function_call_to_filters(func_args)
                print(f"Extracted filters: {filters}")
                return filters
            
            return {}
            
        except Exception as e:
            print(f"Filter extraction error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {}

def search_employees(query_embedding, filters, limit=1000):
    supabase = get_supabase()
    query_builder = supabase.table("employees").select("*, employee_embeddings(embedding)")

    # Handle date filters - can have both after and before for range queries
    if "join_date" in filters:
        print(f"Filtering by exact join_date: {filters['join_date']}")
        query_builder = query_builder.eq("join_date", filters["join_date"])
    else:
        if "join_date_after" in filters:
            print(f"Filtering join_date > {filters['join_date_after']}")
            query_builder = query_builder.gt("join_date", filters["join_date_after"])
        if "join_date_before" in filters:
            print(f"Filtering join_date < {filters['join_date_before']}")
            query_builder = query_builder.lt("join_date", filters["join_date_before"])

    if "department" in filters:
        print(f"Filtering by department: {filters['department']}")
        query_builder = query_builder.eq("department", filters["department"])

    if "exact_experience" in filters:
        print(f"Filtering exact_experience = {filters['exact_experience']}")
        query_builder = query_builder.eq("experience_years", filters["exact_experience"])
    else:
        if "min_experience" in filters:
            print(f"Filtering min_experience >= {filters['min_experience']}")
            query_builder = query_builder.gte("experience_years", filters["min_experience"])
        if "max_experience" in filters:
            print(f"Filtering max_experience <= {filters['max_experience']}")
            query_builder = query_builder.lte("experience_years", filters["max_experience"])

    employees_with_embeddings = query_builder.execute().data
    print(f"Database returned {len(employees_with_embeddings)} employees after filters")
    
    if "skills" in filters and filters["skills"]:
        employees_with_embeddings = [
            emp for emp in employees_with_embeddings
            if any(skill.lower() in [s.lower() for s in emp.get("skills", [])] for skill in filters["skills"])
        ]
        if not employees_with_embeddings:
            return []
    
    scored_employees = []
    for emp in employees_with_embeddings:
        emp_embedding = None
        
        if "employee_embeddings" in emp:
            emb_data = (emp["employee_embeddings"][0].get("embedding") 
                       if isinstance(emp["employee_embeddings"], list) and emp["employee_embeddings"]
                       else emp["employee_embeddings"].get("embedding") if isinstance(emp["employee_embeddings"], dict)
                       else None)
            
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
    
    if "sort_by" in filters:
        sort_key = {
            "experience_desc": lambda x: (-x[1]["experience_years"], x[0]),
            "experience_asc": lambda x: (x[1]["experience_years"], x[0]),
            "join_date_desc": lambda x: (-ord(x[1]["join_date"][0]), x[0])
        }.get(filters["sort_by"], lambda x: -x[0])
        scored_employees.sort(key=sort_key)
    else:
        scored_employees.sort(key=lambda x: -x[0])
    
    return [emp for _, emp in scored_employees[:limit]]

def generate_summary(query, employees, max_retries=3):
    if not employees:
        return "No employees found matching your criteria."

    top_employees = employees[:5]
    emp_text = "\n".join([
        f"- {emp['name']}: {', '.join(emp['skills'][:4])} | {emp['department']} | {emp['experience_years']} years | {emp['join_date']}"
        for emp in top_employees
    ])

    for attempt in range(max_retries):
        try:
            response = get_groq_client().chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize employee search results in 1-2 sentences. State EXACT number found. Mention key skills/departments from TOP results only. Keep factual and brief."
                    },
                    {
                        "role": "user",
                        "content": f"Query: '{query}'\nTotal: {len(employees)}\nTop 5:\n{emp_text}"
                    }
                ],
                model=config.LLM_MODEL,
                temperature=0,
                max_tokens=100
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Summary generation error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"Found {len(employees)} employees matching '{query}'."

@app.post("/api/search")
async def search(request: SearchRequest):
    try:
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Search query cannot be empty")

        print(f"Search query: {request.query}")

        # Extract filters with retry logic
        filters = extract_filters(request.query)
        print(f"Extracted filters: {filters}")

        # Create embedding
        query_embedding = create_simple_embedding(request.query)

        # Search employees
        try:
            employees = search_employees(query_embedding, filters)
        except Exception as e:
            print(f"Database search error: {e}")
            raise HTTPException(status_code=503, detail=f"Database connection error. Please try again.")

        # Generate summary with retry logic
        summary = generate_summary(request.query, employees)

        clean_employees = [{
            "id": emp["id"],
            "name": emp["name"],
            "skills": emp["skills"],
            "department": emp["department"],
            "join_date": emp["join_date"],
            "experience_years": emp["experience_years"],
            "bio": emp["bio"],
            "email": emp["email"]
        } for emp in employees]

        print(f"Search completed: {len(clean_employees)} results")

        return {
            "success": True,
            "employees": clean_employees,
            "summary": summary,
            "count": len(clean_employees),
            "filters_applied": filters
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

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
    text = f"""Name: {employee_data['name']}
    Skills: {', '.join(employee_data['skills'])}
    Department: {employee_data['department']}
    Experience: {employee_data['experience_years']} years
    Bio: {employee_data['bio']}
    Joined: {employee_data['join_date']}"""
    return create_simple_embedding(text)

@app.post("/api/admin/add_employee")
async def add_employee(employee: EmployeeCreate):
    try:
        supabase = get_supabase()
        result = supabase.table("employees").select("id").order("id", desc=True).limit(1).execute()
        next_id = (result.data[0]["id"] + 1) if result.data else 1
        
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
        
        embedding = create_employee_embedding(emp_data)
        supabase.table("employee_embeddings").insert({
            "id": next_id,
            "embedding": embedding
        }).execute()
        
        return {"success": True, "message": f"Employee {employee.name} added", "id": next_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.put("/api/admin/update_employee/{employee_id}")
async def update_employee(employee_id: int, employee: EmployeeCreate):
    try:
        supabase = get_supabase()
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
        
        emp_data["id"] = employee_id
        embedding = create_employee_embedding(emp_data)
        supabase.table("employee_embeddings").update({"embedding": embedding}).eq("id", employee_id).execute()
        
        return {"success": True, "message": f"Employee {employee.name} updated"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/admin/delete_employee/{employee_id}")
async def delete_employee(employee_id: int):
    try:
        supabase = get_supabase()
        supabase.table("employee_embeddings").delete().eq("id", employee_id).execute()
        supabase.table("employees").delete().eq("id", employee_id).execute()
        return {"success": True, "message": f"Employee {employee_id} deleted"}
    except Exception as e:
        return {"success": False, "error": str(e)}

class WebhookPayload(BaseModel):
    employee_id: int
    action: str

@app.post("/api/admin/webhook/re-embed")
async def webhook_re_embed(payload: WebhookPayload):
    try:
        supabase = get_supabase()
        result = supabase.table("employees").select("*").eq("id", payload.employee_id).execute()
        if not result.data:
            return {"success": False, "error": "Employee not found"}
        
        embedding = create_employee_embedding(result.data[0])
        supabase.table("employee_embeddings").upsert({
            "id": payload.employee_id,
            "embedding": embedding
        }).execute()
        
        return {"success": True, "message": f"Re-embedded employee {payload.employee_id}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/admin/re-embed-all")
@app.post("/api/admin/re-embed-all")
async def re_embed_all():
    try:
        supabase = get_supabase()
        employees = supabase.table("employees").select("*").execute().data
        
        if not employees:
            return {"success": False, "error": "No employees found"}
        
        success_count = 0
        for employee in employees:
            try:
                embedding = create_employee_embedding(employee)
                supabase.table("employee_embeddings").upsert({
                    "id": employee["id"],
                    "embedding": embedding
                }).execute()
                success_count += 1
            except Exception as e:
                print(f"Error embedding employee {employee['id']}: {e}")
        
        return {
            "success": True,
            "message": f"Re-embedded {success_count}/{len(employees)} employees",
            "total": len(employees),
            "success_count": success_count
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
