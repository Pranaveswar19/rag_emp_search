-- Supabase Trigger for Automatic Re-embedding
-- Run this in Supabase SQL Editor
-- Replace YOUR-RAILWAY-URL with your actual Railway app URL

-- Step 1: Create a webhook function that calls Railway API when employee changes
CREATE OR REPLACE FUNCTION notify_employee_change()
RETURNS TRIGGER AS $$
BEGIN
  -- Call Railway endpoint to re-embed with HIGH QUALITY embeddings
  PERFORM net.http_post(
    url := 'https://YOUR-RAILWAY-URL.railway.app/api/admin/webhook/re-embed',
    headers := '{"Content-Type": "application/json"}'::jsonb,
    body := json_build_object(
      'employee_id', NEW.id,
      'action', TG_OP
    )::text
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Step 2: Create triggers for INSERT and UPDATE
DROP TRIGGER IF EXISTS employee_insert_trigger ON employees;
CREATE TRIGGER employee_insert_trigger
  AFTER INSERT ON employees
  FOR EACH ROW
  EXECUTE FUNCTION notify_employee_change();

DROP TRIGGER IF EXISTS employee_update_trigger ON employees;
CREATE TRIGGER employee_update_trigger
  AFTER UPDATE ON employees
  FOR EACH ROW
  EXECUTE FUNCTION notify_employee_change();
