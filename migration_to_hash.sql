-- Migration: Revert from OpenAI embeddings (1536-dim) to hash embeddings (384-dim)
-- This makes the project 100% FREE (no paid APIs)

-- Step 1: Drop existing embeddings table
DROP TABLE IF EXISTS employee_embeddings CASCADE;

-- Step 2: Recreate with 384 dimensions for hash-based embeddings
CREATE TABLE employee_embeddings (
    id INT PRIMARY KEY REFERENCES employees(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL
);

-- Step 3: Create index for fast similarity search
CREATE INDEX ON employee_embeddings USING ivfflat (embedding vector_cosine_ops);

-- You'll need to re-embed all employees after this migration
-- Visit: https://your-app-url/api/admin/re-embed-all
