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

FILTER_SCHEMA = {
    "skills": {
        "type": "array",
        "description": "Specific programming languages, frameworks, or technologies",
        "examples": ["Python", "React", "AWS", "Docker"],
        "blacklist": {
            "developers", "developer", "engineers", "engineer", "backend", "frontend",
            "full-stack", "fullstack", "senior", "junior", "staff", "people", "expert",
            "specialist", "lead", "principal", "architect", "manager", "programmers",
            "programmer", "coders", "coder"
        }
    },
    "department": {
        "type": "string",
        "values": ["Engineering", "Data", "Design", "Product", "Marketing", "Sales", 
                   "HR", "Finance", "Security", "QA", "Operations", "Legal", 
                   "Business Development", "Customer Success"]
    },
    "exact_experience": {
        "type": "integer",
        "keywords": ["years", "exactly", "with X years", "X years experience"],
        "examples": ["5 years experience", "exactly 3 years", "with 7 years"]
    },
    "min_experience": {
        "type": "integer",
        "keywords": ["at least", "or more", "or higher", "+", "minimum"],
        "exclusive_keywords": ["above", "over"],
        "offset": 1,
        "examples": ["5+ years", "at least 5 years", "above 5 years → 6"]
    },
    "max_experience": {
        "type": "integer",
        "keywords": ["or less", "or lower", "or fewer", "up to", "maximum"],
        "exclusive_keywords": ["below", "under", "less than"],
        "offset": -1,
        "examples": ["5 or less", "up to 5 years", "below 5 years → 4"]
    },
    "join_date": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "keywords": ["joined on", "hired on"]
    },
    "join_date_after": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "keywords": ["joined after", "hired after", "since"]
    },
    "join_date_before": {
        "type": "date",
        "format": "YYYY-MM-DD",
        "keywords": ["joined before", "hired before", "prior to"]
    },
    "sort_by": {
        "type": "string",
        "values": {
            "experience_desc": ["most experienced", "most years", "highest experience"],
            "experience_asc": ["least experienced", "newest", "lowest experience"],
            "join_date_desc": ["most recent", "latest hires", "recently joined"]
        }
    }
}

FEW_SHOT_EXAMPLES = [
    ("Python developers", {"skills": ["Python"]}),
    ("Backend developers in Engineering department", {"department": "Engineering"}),
    ("React developers with 5 years experience", {"skills": ["React"], "exact_experience": 5}),
    ("Developers with 5+ years experience", {"min_experience": 5}),
    ("Python devs with 5 years of experience or lower", {"skills": ["Python"], "max_experience": 5}),
    ("Engineers with less than 3 years experience", {"max_experience": 2}),
    ("People with above 10 years experience", {"min_experience": 11}),
    ("Developers between 3 and 7 years experience", {"min_experience": 3, "max_experience": 7}),
    ("Employees who joined on 2023-09-15", {"join_date": "2023-09-15"}),
    ("Employees who joined before 2024", {"join_date_before": "2024-01-01"}),
    ("People who joined after 2023", {"join_date_after": "2023-12-31"}),
    ("Employees who joined in 2024", {"join_date_after": "2024-01-01", "join_date_before": "2024-12-31"}),
    ("Person with most years of experience", {"sort_by": "experience_desc"})
]

def build_filter_prompt():
    available_fields = []
    for field, schema in FILTER_SCHEMA.items():
        field_type = schema["type"]
        desc = schema.get("description", "")
        if "keywords" in schema:
            keywords = ", ".join(f'"{k}"' for k in schema["keywords"])
            available_fields.append(f"- {field}: {field_type} (use for: {keywords})")
        else:
            available_fields.append(f"- {field}: {field_type}")
    
    rules = [
        "EXPERIENCE RULES:",
        '- "X years" → exact_experience: X',
        '- "X+" or "at least X" → min_experience: X',
        '- "X or less" → max_experience: X',
        '- "below X" or "under X" → max_experience: X-1',
        '- "above X" or "over X" → min_experience: X+1',
        '- "between X and Y" → min_experience: X, max_experience: Y',
        '- When ambiguous, prefer exact_experience',
        "",
        "DATE RULES:",
        '- "joined after YYYY" → join_date_after: "YYYY-12-31"',
        '- "joined before YYYY" → join_date_before: "YYYY-01-01"',
        '- "joined in YYYY" → both join_date_after and join_date_before',
        "",
        "GENERAL RULES:",
        '- Skills: ONLY technology names (ignore role descriptions)',
        '- "most experienced" → sort_by, NOT min_experience',
        f'- Today: {datetime.now().strftime("%Y-%m-%d")}'
    ]
    
    return "Extract filters from employee search queries. Return ONLY a JSON object.\n\nAvailable fields:\n" + \
           "\n".join(available_fields) + "\n\n" + "\n".join(rules)

def validate_and_clean_filters(filters):
    cleaned = {}
    
    for field, value in filters.items():
        if field not in FILTER_SCHEMA:
            continue
            
        schema = FILTER_SCHEMA[field]
        
        if schema["type"] == "array" and isinstance(value, list):
            if "blacklist" in schema:
                value = [v for v in value if v.lower() not in schema["blacklist"]]
            if value:
                cleaned[field] = value
                
        elif schema["type"] == "string":
            if "values" in schema and isinstance(schema["values"], dict):
                cleaned[field] = value
            elif "values" in schema and value in schema["values"]:
                cleaned[field] = value
            else:
                cleaned[field] = value
                
        elif schema["type"] == "integer":
            try:
                cleaned[field] = int(value)
            except (ValueError, TypeError):
                print(f"Invalid integer for {field}: {value}")
                
        elif schema["type"] == "date":
            try:
                parts = value.split('-')
                if len(parts) == 3 and all(p.isdigit() for p in parts):
                    cleaned[field] = value
            except:
                print(f"Invalid date for {field}: {value}")
    
    return cleaned

def extract_filters(query, max_retries=3):
    for attempt in range(max_retries):
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
- exact_experience: integer for exact years (use for "X years", "exactly X years", "with X years experience")
- min_experience: integer for minimum years (use for "X+ years", "at least X years", "X or more years")
- max_experience: integer for maximum years (use for "X or less", "X or fewer", "X or lower", "under X years", "less than X years")
- sort_by: string - either "experience_desc" (most experienced), "experience_asc" (least experienced), or "join_date_desc" (most recent)

CRITICAL RULES FOR EXPERIENCE FILTERING:
- "5 years experience" → exact_experience: 5
- "5+ years" or "at least 5 years" → min_experience: 5
- "5 or lower" or "5 or less" → max_experience: 5
- "below 5 years" or "under 5 years" → max_experience: 4
- "above 5 years" or "over 5 years" → min_experience: 6
- "between 3 and 7 years" → min_experience: 3, max_experience: 7
- When ambiguous, prefer exact_experience
- NEVER use min_experience for phrases with "or less", "or lower", "below", "under"
- NEVER use max_experience for phrases with "or more", "or higher", "above", "over"

CRITICAL RULES FOR DATE FILTERING:
- "joined after YYYY" (year only) → use "join_date_after": "YYYY-12-31" (NOT "YYYY-01-01")
- "joined after YYYY-MM-DD" (full date) → use "join_date_after": "YYYY-MM-DD"
- "joined before YYYY" (year only) → use "join_date_before": "YYYY-01-01"
- "joined before YYYY-MM-DD" (full date) → use "join_date_before": "YYYY-MM-DD"
- "joined in YYYY" → use both "join_date_after": "YYYY-01-01" AND "join_date_before": "YYYY-12-31" (keep both separate)

OTHER IMPORTANT RULES:
- "skills" should contain ONLY technology names. Ignore words like: developers, engineers, Backend, Frontend, Full-stack, Senior, Junior, people, staff.
- If query says "joined on [date]" use "join_date", NOT "join_date_after"
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
                    "content": "Python devs with 5 years of experience or lower"
                },
                {
                    "role": "assistant",
                    "content": '{"skills": ["Python"], "max_experience": 5}'
                },
                {
                    "role": "user",
                    "content": "Engineers with less than 3 years experience"
                },
                {
                    "role": "assistant",
                    "content": '{"max_experience": 2}'
                },
                {
                    "role": "user",
                    "content": "People with above 10 years experience"
                },
                {
                    "role": "assistant",
                    "content": '{"min_experience": 11}'
                },
                {
                    "role": "user",
                    "content": "Developers between 3 and 7 years experience"
                },
                {
                    "role": "assistant",
                    "content": '{"min_experience": 3, "max_experience": 7}'
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
                    "content": "People who joined after 2023"
                },
                {
                    "role": "assistant",
                    "content": '{"join_date_after": "2023-12-31"}'
                },
                {
                    "role": "user",
                    "content": "Employees who joined in 2024"
                },
                {
                    "role": "assistant",
                    "content": '{"join_date_after": "2024-01-01", "join_date_before": "2024-12-31"}'
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
            max_tokens=200
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

            # Validate date formats
            for date_key in ["join_date", "join_date_after", "join_date_before"]:
                if date_key in filters:
                    try:
                        # Validate date format YYYY-MM-DD
                        date_parts = filters[date_key].split('-')
                        if len(date_parts) != 3:
                            print(f"Invalid date format for {date_key}: {filters[date_key]}")
                            del filters[date_key]
                    except:
                        print(f"Error validating {date_key}: {filters.get(date_key)}")
                        if date_key in filters:
                            del filters[date_key]
            
            # Validate experience values
            for exp_key in ["exact_experience", "min_experience", "max_experience"]:
                if exp_key in filters:
                    try:
                        filters[exp_key] = int(filters[exp_key])
                    except (ValueError, TypeError):
                        print(f"Invalid experience value for {exp_key}: {filters[exp_key]}")
                        del filters[exp_key]

            return filters
        except Exception as e:
            print(f"Filter extraction error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                # If all retries fail, return empty filters instead of breaking
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
