-- Function to create simple hash-based embeddings directly in PostgreSQL
CREATE OR REPLACE FUNCTION create_simple_embedding(text_input TEXT, dimension INTEGER DEFAULT 384)
RETURNS vector AS $$
DECLARE
    words TEXT[];
    embedding FLOAT[];
    word TEXT;
    word_index INTEGER := 0;
    hash_val INTEGER;
    norm FLOAT := 0;
    i INTEGER;
BEGIN
    -- Initialize embedding array with zeros
    embedding := ARRAY_FILL(0::FLOAT, ARRAY[dimension]);
    
    -- Split text into words and convert to lowercase
    words := regexp_split_to_array(lower(text_input), '\s+');
    
    -- Process each word
    FOREACH word IN ARRAY words
    LOOP
        word_index := word_index + 1;
        -- Create hash value (using hashtext which is built-in)
        hash_val := (hashtext(word) % dimension);
        -- Make sure hash_val is positive
        IF hash_val < 0 THEN
            hash_val := hash_val + dimension;
        END IF;
        -- Add to embedding (array indices in PostgreSQL start at 1)
        embedding[hash_val + 1] := embedding[hash_val + 1] + (1.0 / word_index);
    END LOOP;
    
    -- Calculate L2 norm
    FOR i IN 1..dimension LOOP
        norm := norm + (embedding[i] * embedding[i]);
    END LOOP;
    norm := sqrt(norm);
    
    -- Normalize embedding if norm > 0
    IF norm > 0 THEN
        FOR i IN 1..dimension LOOP
            embedding[i] := embedding[i] / norm;
        END LOOP;
    END IF;
    
    -- Convert to vector type
    RETURN embedding::vector;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to create employee text representation
CREATE OR REPLACE FUNCTION create_employee_text(emp employees)
RETURNS TEXT AS $$
BEGIN
    RETURN format(
        'Name: %s
    Skills: %s
    Department: %s
    Experience: %s years
    Bio: %s
    Joined: %s',
        emp.name,
        array_to_string(emp.skills, ', '),
        emp.department,
        emp.experience_years,
        emp.bio,
        emp.join_date
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger function to auto-generate embeddings
CREATE OR REPLACE FUNCTION auto_generate_embedding()
RETURNS TRIGGER AS $$
DECLARE
    employee_text TEXT;
    new_embedding vector(384);
BEGIN
    -- Create text representation of employee
    employee_text := create_employee_text(NEW);
    
    -- Generate embedding
    new_embedding := create_simple_embedding(employee_text, 384);
    
    -- Insert or update embedding
    INSERT INTO employee_embeddings (id, embedding)
    VALUES (NEW.id, new_embedding)
    ON CONFLICT (id) 
    DO UPDATE SET embedding = new_embedding, created_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS employee_auto_embed_insert ON employees;
DROP TRIGGER IF EXISTS employee_auto_embed_update ON employees;

-- Create triggers for INSERT and UPDATE
CREATE TRIGGER employee_auto_embed_insert
    AFTER INSERT ON employees
    FOR EACH ROW
    EXECUTE FUNCTION auto_generate_embedding();

CREATE TRIGGER employee_auto_embed_update
    AFTER UPDATE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION auto_generate_embedding();

-- Test the function (optional - you can remove this)
-- SELECT create_simple_embedding('Python developer with 5 years experience', 384);
