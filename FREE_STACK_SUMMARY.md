# 100% Free RAG Employee Search - Cost Breakdown

## ✅ Current Stack (All FREE)

| Component | Service | Cost | Notes |
|-----------|---------|------|-------|
| **LLM** | Groq (Llama 3.1) | $0/month | Free API, fast inference |
| **Embeddings** | Hash-based (local) | $0/month | No external API calls |
| **Database** | Supabase Free Tier | $0/month | 500MB storage, 2GB bandwidth |
| **Hosting** | Railway/Render Free | $0/month | Free tier available |
| **Vector Search** | pgvector (Supabase) | $0/month | Included in Supabase |

**Total: $0/month** 🎉

## ❌ What We Removed

| Component | Service | Cost | Why Removed |
|-----------|---------|------|-------------|
| **Embeddings** | OpenAI API | ~$0.10/month | Replaced with free hash-based |

## How It Works Now

1. **Query Processing**: Groq LLM (free) extracts filters from natural language
2. **Embedding**: Hash-based algorithm creates 384-dim vectors locally (no API)
3. **Search**: Supabase pgvector performs similarity search (free tier)
4. **Summary**: Groq LLM (free) generates human-readable summary

## Performance Comparison

### OpenAI Embeddings (Removed)
- ✅ High quality semantic understanding
- ❌ Costs money ($0.00002 per 1K tokens)
- ❌ Network latency for API calls
- ❌ Requires credit card

### Hash-Based Embeddings (Current)
- ✅ 100% free
- ✅ No network latency (runs locally)
- ✅ Fast computation
- ✅ Works on free hosting tiers
- ⚠️ Simpler than ML models (but still effective with LLM filters)

## Migration Instructions

See `MIGRATION_TO_FREE.md` for detailed steps.

Quick start:
1. Run `migration_to_hash.sql` in Supabase SQL Editor
2. Remove `OPENAI_API_KEY` from environment variables
3. Deploy updated code
4. Visit `/api/admin/re-embed-all` to populate embeddings

## Free Service Limits

### Supabase Free Tier
- 500MB database storage
- 2GB bandwidth/month
- 50,000 monthly active users
- Unlimited API requests

### Groq Free Tier
- Rate limits apply (check groq.com)
- Fast inference on Llama models
- No credit card required

### Railway/Render Free Tier
- 500 hours/month (Railway)
- Auto-sleep after inactivity (Render)
- Sufficient for demo/small projects

## When to Upgrade

Consider paid services if you need:
- Higher quality semantic search (OpenAI/Cohere embeddings)
- More database storage (>500MB)
- Higher traffic (>2GB bandwidth/month)
- Always-on hosting without sleep

For most small-to-medium projects, the free tier is sufficient!
