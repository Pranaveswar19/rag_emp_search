-- Migration: Update embedding dimension from 384 to 1536 for OpenAI embeddings
-- Run this in Supabase SQL Editor BEFORE deploying the new code

-- Drop existing index
DROP INDEX IF EXISTS employee_embeddings_idx;

-- Drop and recreate the employee_embeddings table with new dimension
DROP TABLE IF EXISTS employee_embeddings;

CREATE TABLE employee_embeddings (
    id INTEGER PRIMARY KEY REFERENCES employees(id),
    embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
    created_at TIMESTAMP DEFAULT NOW()
);

-- Recreate index with new dimension
CREATE INDEX employee_embeddings_idx ON employee_embeddings USING ivfflat (embedding vector_cosine_ops);

-- Note: You'll need to re-embed all employees after this migration
