import sqlite3
import os

# Delete old database
if os.path.exists("users.db"):
    try:
        os.remove("users.db")
        print("✓ Old database deleted")
    except Exception as e:
        print(f"✗ Could not delete old database: {e}")
        print("  Please stop the server and run this script again")
        exit(1)

# Import and create new database with updated schema
from main import Base, engine

Base.metadata.create_all(bind=engine)
print("✓ New database created with updated schema")

# Verify the schema
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(training_jobs)")
columns = cursor.fetchall()

print("\nDatabase columns:")
for col in columns:
    print(f"  - {col[1]:20s} ({col[2]})")

# Check for problem_type
column_names = [col[1] for col in columns]
if 'problem_type' in column_names:
    print("\n✓ SUCCESS: Database migration complete! 'problem_type' column added.")
else:
    print("\n✗ ERROR: 'problem_type' column is still missing!")

conn.close()
