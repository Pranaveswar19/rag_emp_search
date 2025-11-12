DROP TABLE IF EXISTS employee_embeddings CASCADE;

CREATE TABLE employee_embeddings (
    id INT PRIMARY KEY REFERENCES employees(id) ON DELETE CASCADE,
    embedding vector(384) NOT NULL
);

CREATE INDEX ON employee_embeddings USING ivfflat (embedding vector_cosine_ops);
