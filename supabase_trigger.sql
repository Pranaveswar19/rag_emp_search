CREATE OR REPLACE FUNCTION notify_employee_change()
RETURNS TRIGGER AS $$
BEGIN
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
