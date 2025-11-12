# RAG Employee Search System - Technical Documentation

**Project:** RAG-based Employee Search System
**Version:** 1.0
**Date:** January 2025
**Repository:** https://github.com/Pranaveswar19/rag_emp_search

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Technology Stack](#technology-stack)
4. [System Architecture](#system-architecture)
5. [Data Pipeline](#data-pipeline)
6. [Implementation Process](#implementation-process)
7. [Key Features](#key-features)
8. [API Documentation](#api-documentation)
9. [Deployment Strategy](#deployment-strategy)
10. [Roadmap](#roadmap)
11. [Cost Analysis](#cost-analysis)

---

## Executive Summary

The RAG Employee Search System is a 100% free, AI-powered employee search platform that enables natural language queries to find and filter employees across an organization. The system leverages Retrieval-Augmented Generation (RAG) to provide intelligent search capabilities without requiring expensive embedding APIs or infrastructure.

**Key Achievements:**
- 100% free stack with zero API costs
- Natural language query processing
- Intelligent filter extraction
- Real-time search with semantic understanding
- Scalable architecture ready for production

---

## Project Overview

### Problem Statement
Traditional employee databases require users to know exact search parameters (SQL queries, specific filters). Non-technical users struggle to find employees based on natural language descriptions like "Python developers with 5+ years in Engineering department."

### Solution
A RAG-based search system that:
1. Accepts natural language queries
2. Extracts structured filters using LLM
3. Performs vector similarity search with hash-based embeddings
4. Returns ranked results with AI-generated summaries

### Target Users
- HR teams looking for candidates
- Managers searching for team members with specific skills
- Recruiters finding talent within the organization
- Employees seeking collaborators

---

## Technology Stack

### Backend Framework
- **FastAPI** (Python)
  - High-performance async web framework
  - Automatic API documentation (Swagger/OpenAPI)
  - Native support for async/await patterns
  - Type hints and validation with Pydantic

### Database & Vector Store
- **Supabase** (PostgreSQL + pgvector)
  - Free tier: 500MB database, 2GB bandwidth
  - Built-in authentication and row-level security
  - Real-time subscriptions
  - RESTful API auto-generation
  - pgvector extension for vector similarity search

### Language Model
- **Groq** (Llama 3.1-8B-Instant)
  - Free tier: 14,400 requests/day, 30 requests/minute
  - Ultra-fast inference (~300 tokens/second)
  - Used for:
    - Filter extraction from natural language
    - Summary generation for search results

### Embeddings
- **Hash-based Embeddings** (Custom Implementation)
  - 100% free, runs locally
  - No API calls required
  - 384-dimensional vectors
  - Position-weighted word hashing
  - L2 normalization for similarity comparison

### Hosting
- **Railway / Render**
  - Free tier available
  - Automatic deployments from GitHub
  - Custom domains supported
  - Environment variable management

### Frontend
- **Vanilla JavaScript + HTML/CSS**
  - No framework dependencies
  - Fast load times
  - Responsive design
  - Real-time search feedback

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Interface                       │
│                    (HTML + JavaScript)                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  │ HTTP/REST
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Search     │  │   Filter     │  │   Summary    │      │
│  │  Endpoint    │  │  Extraction  │  │  Generation  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         │                  ▼                  ▼              │
│         │          ┌──────────────────────────────┐         │
│         │          │     Groq LLM API             │         │
│         │          │   (Llama 3.1-8B-Instant)     │         │
│         │          └──────────────────────────────┘         │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────┐          │
│  │      Hash-based Embedding Generator          │          │
│  │  (384-dim vectors, position-weighted)        │          │
│  └──────────────────────────────────────────────┘          │
│                       │                                      │
└───────────────────────┼──────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Supabase (PostgreSQL)                      │
│  ┌──────────────────┐       ┌──────────────────┐           │
│  │   employees      │       │ employee_embeddings│          │
│  │  - id            │       │  - id              │          │
│  │  - name          │       │  - embedding       │          │
│  │  - skills        │       │    (vector[384])   │          │
│  │  - department    │       └────────────────────┘          │
│  │  - join_date     │                                        │
│  │  - experience    │                                        │
│  │  - bio           │                                        │
│  │  - email         │                                        │
│  └──────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Frontend Layer
- Single-page application
- Asynchronous fetch API calls
- Real-time error handling
- Console logging for debugging

#### 2. API Layer (FastAPI)
- RESTful endpoints
- Request validation
- Error handling with retry logic
- Rate limit management

#### 3. LLM Integration (Groq)
- Filter extraction from natural language
- Result summarization
- Exponential backoff for rate limits
- Graceful degradation on failures

#### 4. Embedding Engine
- Hash-based vector generation
- No external API dependencies
- Consistent reproducibility
- Fast computation

#### 5. Database Layer (Supabase)
- Relational data storage
- Vector similarity operations
- Filtered queries
- Real-time capabilities

---

## Data Pipeline

### 1. Employee Data Ingestion

```
Employee Data (JSON)
        ↓
Validation & Transformation
        ↓
┌───────────────────────────┐
│  Insert into employees    │
│  table                    │
└───────────────────────────┘
        ↓
Generate Text Representation
        ↓
Hash-based Embedding (384-dim)
        ↓
┌───────────────────────────┐
│  Insert into              │
│  employee_embeddings      │
│  table                    │
└───────────────────────────┘
```

### 2. Search Query Pipeline

```
User Query: "Python developers with 5+ years"
        ↓
┌─────────────────────────────────────────────┐
│  Step 1: Filter Extraction (Groq LLM)       │
│  Output: {                                   │
│    "skills": ["Python"],                     │
│    "min_experience": 5                       │
│  }                                           │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Step 2: Query Embedding Generation         │
│  Hash-based embedding of query text         │
│  Output: [0.12, 0.45, ..., 0.33] (384-dim)  │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Step 3: Database Filtering                 │
│  - Apply structured filters (skills, exp)   │
│  - Retrieve employee embeddings             │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Step 4: Similarity Scoring                 │
│  - Compute dot product with query embedding │
│  - Rank by similarity score                 │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│  Step 5: Summary Generation (Groq LLM)      │
│  Input: Top 5 results                       │
│  Output: "Found 12 Python developers..."    │
└─────────────────────────────────────────────┘
        ↓
Return Results to User
```

### 3. Re-embedding Pipeline (Webhooks)

```
Employee Update Event
        ↓
Webhook Trigger
        ↓
Fetch Updated Employee Data
        ↓
Regenerate Embedding
        ↓
Update employee_embeddings table
        ↓
Complete
```

---



## Key Features

### 1. Natural Language Processing
- **Smart Filter Extraction**: Converts queries like "React developers with 5+ years" into structured filters
- **Context Understanding**: Interprets dates, experience levels, and skills
- **Flexible Queries**: Supports various query formats

### 2. Advanced Search Capabilities
- **Skills Filtering**: Array-based skill matching
- **Experience Range**: Min/max experience filters
- **Department Filtering**: Exact match on departments
- **Date Range Queries**: Join date filtering with year/date support
- **Sorting Options**: By experience, join date, or relevance

### 3. Error Handling & Reliability
- **Retry Logic**: Automatic retries with exponential backoff
- **Rate Limit Management**: Graceful handling of API limits
- **Fallback Mechanisms**: Degrades gracefully on LLM failures
- **Detailed Logging**: Console and server logs for debugging

### 4. User Experience
- **Real-time Search**: Instant results as you type
- **Example Queries**: Pre-built examples for users
- **Result Summaries**: AI-generated summaries of search results
- **Clean Interface**: Responsive, modern design

---

## API Documentation

### Search Endpoint

**POST** `/api/search`

**Request Body:**
```json
{
  "query": "Python developers with 5+ years in Engineering"
}
```

**Response:**
```json
{
  "success": true,
  "count": 12,
  "employees": [
    {
      "id": 1,
      "name": "Sarah Johnson",
      "skills": ["Python", "React", "PostgreSQL"],
      "department": "Engineering",
      "experience_years": 5,
      "join_date": "2024-03-15",
      "bio": "Full-stack developer...",
      "email": "sarah.j@orants.ai"
    }
  ],
  "summary": "Found 12 Python developers in Engineering with 5+ years experience...",
  "filters_applied": {
    "skills": ["Python"],
    "min_experience": 5,
    "department": "Engineering"
  }
}
```

### Admin Endpoints

**POST** `/api/admin/add_employee`
- Adds new employee
- Auto-generates embeddings

**PUT** `/api/admin/update_employee/{id}`
- Updates employee data
- Regenerates embeddings

**DELETE** `/api/admin/delete_employee/{id}`
- Removes employee and embeddings

**POST** `/api/admin/re-embed-all`
- Regenerates all embeddings
- Useful after embedding algorithm changes

---

## Deployment Strategy

### Environment Setup
```bash
# Required Environment Variables
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_anon_key
GROQ_API_KEY=your_groq_api_key
```

### Deployment Steps

#### Option 1: Railway
1. Connect GitHub repository
2. Configure environment variables
3. Deploy automatically on push

#### Option 2: Render
1. Create Web Service from repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Configure environment variables

### Database Setup
1. Create Supabase project
2. Run `setup_database.sql` in SQL Editor
3. Populate initial data: `python embed_employees.py`

---


## Cost Analysis

### Current Stack (100% Free)

| Service | Free Tier Limits | Usage | Cost |
|---------|------------------|-------|------|
| **Supabase** | 500MB DB, 2GB bandwidth, 50K monthly active users | ~100 employees, moderate traffic | $0 |
| **Groq** | 14,400 req/day, 30 req/min | ~1000 searches/day | $0 |
| **Railway/Render** | 500 hours/month, 512MB RAM | Single service | $0 |
| **Hash Embeddings** | Local computation | All employees | $0 |
| **Domain (Optional)** | - | Custom domain | ~$12/year |

**Total Monthly Cost: $0** (excluding optional domain)

### Scale Projections

#### 1,000 Employees, 10,000 Searches/Month
| Service | Requirement | Free Tier? | Estimated Cost |
|---------|-------------|------------|----------------|
| Supabase | 2GB DB, 10GB bandwidth | ❌ Upgrade needed | $25/month (Pro) |
| Groq | 20,000 req/month | ✅ Still free | $0 |
| Hosting | 1GB RAM | ❌ Upgrade needed | $5-10/month |
| **Total** | | | **$30-35/month** |

#### 10,000 Employees, 100,000 Searches/Month
| Service | Requirement | Free Tier? | Estimated Cost |
|---------|-------------|------------|----------------|
| Supabase | 10GB DB, 50GB bandwidth | ❌ Team tier | $599/month |
| Groq | 200,000 req/month | ❌ May need paid tier | TBD |
| Hosting | 4GB RAM, load balancing | ❌ | $50-100/month |
| **Total** | | | **$650-700/month** |

### Alternative: Paid Embeddings (OpenAI)
If switching to OpenAI embeddings (text-embedding-3-small):
- **Cost**: $0.02 per 1M tokens
- **10,000 employees**: ~$0.50 one-time
- **100,000 searches/month**: ~$5/month
- **Better quality** but removes "100% free" advantage

---

## Technical Challenges & Solutions

### Challenge 1: LLM Rate Limits
**Problem**: Groq free tier has 30 requests/minute limit
**Solution**:
- Exponential backoff retry logic
- Request queuing for high traffic
- Fallback to basic keyword search on failure

### Challenge 2: Embedding Quality
**Problem**: Hash-based embeddings less accurate than transformer models
**Solution**:
- Position-weighted hashing for better context
- Hybrid approach: structured filters + semantic search
- Fine-tuned filter extraction compensates

### Challenge 3: Date Query Ambiguity
**Problem**: "after 2023" could mean ≥2023 or >2023
**Solution**:
- Clear LLM prompts with examples
- Date validation and normalization
- Detailed logging for debugging

### Challenge 4: Cold Start Performance
**Problem**: First query slow due to database connection
**Solution**:
- Connection pooling
- Keep-alive connections
- Health check endpoint

---

## Conclusion

The RAG Employee Search System demonstrates that powerful AI-driven search capabilities can be built entirely on free-tier services. By combining intelligent LLM-based filter extraction with hash-based embeddings, the system provides a robust, scalable solution for employee discovery.

The modular architecture allows for easy upgrades (e.g., switching to paid embeddings) as the organization scales, while maintaining the core functionality that makes the system valuable: natural language understanding and semantic search.

---

## Appendices

### A. Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-dotenv==1.0.0
supabase==2.10.0
groq==0.4.0
numpy==1.26.4
pydantic==2.5.0
```

### B. Database Schema

**employees table:**
```sql
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    skills TEXT[] NOT NULL,
    department TEXT NOT NULL,
    join_date DATE NOT NULL,
    experience_years INTEGER NOT NULL,
    bio TEXT,
    email TEXT UNIQUE NOT NULL
);
```

**employee_embeddings table:**
```sql
CREATE TABLE employee_embeddings (
    id INTEGER PRIMARY KEY REFERENCES employees(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL
);
```

### C. References
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Supabase Documentation: https://supabase.com/docs
- Groq API Documentation: https://console.groq.com/docs
- pgvector Documentation: https://github.com/pgvector/pgvector

---

**Document Version**: 1.0
**Last Updated**: January 2025
**Maintained By**: Development Team
**Contact**: https://github.com/Pranaveswar19/rag_emp_search
