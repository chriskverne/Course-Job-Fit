from db_config import get_connection
from dotenv import load_dotenv
import os


load_dotenv()

def execute_schema_sql():
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    
    with open(schema_path, "r") as file:
        schema_sql = file.read()

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(schema_sql)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Schema initialized successfully.")
    except Exception as e:
        print("❌ Error initializing schema:", e)

if __name__ == "__main__":
    execute_schema_sql()
