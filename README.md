# RAG Employee Search

Smart employee search system with natural language queries using RAG (Retrieval-Augmented Generation).

## Features

- Natural language search queries
- Filter by skills, department, experience, join date
- Semantic search with hash-based embeddings
- Smart result sorting and summaries
- Admin APIs for employee management
- Auto re-embedding via webhooks

## Tech Stack

- **Backend**: FastAPI + Python
- **Database**: Supabase (PostgreSQL + pgvector)
- **LLM**: Groq (Llama 3.1) - Free
- **Embeddings**: Hash-based (Local) - Free
- **Hosting**: Railway/Render

## Setup

1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables in `.env`:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   GROQ_API_KEY=your_groq_api_key
   ```
4. Run database setup in Supabase SQL Editor: `setup_database.sql`
5. Populate employees: `python embed_employees.py`
6. Start server: `uvicorn main:app --reload`

## API Endpoints

### Search
- `POST /api/search` - Search employees with natural language

### Admin
- `POST /api/admin/add_employee` - Add new employee
- `PUT /api/admin/update_employee/{id}` - Update employee
- `DELETE /api/admin/delete_employee/{id}` - Delete employee
- `POST /api/admin/re-embed-all` - Re-embed all employees

## Deployment

Deploy to Railway or Render using `render.yaml` configuration.

## Cost

100% Free - No paid APIs required.
