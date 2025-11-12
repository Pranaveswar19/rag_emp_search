# Migration to 100% Free Stack

## Changes Made
Reverted from OpenAI embeddings (paid) to hash-based embeddings (free).

## What's FREE Now
✅ **Groq API** - Free LLM for filter extraction and summaries  
✅ **Hash-based embeddings** - No external API, runs locally  
✅ **Supabase** - Free tier (up to 500MB database, 2GB bandwidth/month)  
✅ **Railway/Render** - Free tier available for hosting  

## What Was REMOVED
❌ **OpenAI API** - Was costing ~$0.10/month for embeddings

## Migration Steps

### 1. Update Supabase Database
Run this SQL in Supabase SQL Editor:
```sql
-- See migration_to_hash.sql
```

### 2. Update Environment Variables
Remove `OPENAI_API_KEY` from your `.env` and hosting platform (Railway/Render).

Required variables now:
- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GROQ_API_KEY` (free at https://console.groq.com)

### 3. Re-embed All Employees
After deploying, visit:
```
POST https://your-app-url/api/admin/re-embed-all
```

Or run locally:
```bash
python embed_employees.py
```

## Performance Notes
- **Quality**: Hash embeddings are simpler than ML models, but combined with Groq's LLM filter extraction, search quality remains good
- **Speed**: Faster than OpenAI API calls (no network latency)
- **Cost**: $0/month 🎉

## Tech Stack (All Free)
- FastAPI (Python web framework)
- Groq API (free LLM - Llama 3.1)
- Supabase (PostgreSQL + pgvector)
- Hash-based embeddings (local computation)
- Railway/Render Free Tier
