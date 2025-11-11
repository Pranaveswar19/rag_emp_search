CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    skills TEXT[] NOT NULL,
    department TEXT NOT NULL,
    join_date DATE NOT NULL,
    experience_years INTEGER NOT NULL,
    bio TEXT,
    email TEXT
);

CREATE TABLE IF NOT EXISTS employee_embeddings (
    id INTEGER PRIMARY KEY REFERENCES employees(id),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS employee_embeddings_idx ON employee_embeddings USING ivfflat (embedding vector_cosine_ops);
