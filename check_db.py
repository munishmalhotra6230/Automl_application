import sqlite3
import os

db_path = "users.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check training_jobs table
    cursor.execute("PRAGMA table_info(training_jobs)")
    columns = cursor.fetchall()
    print("=" * 50)
    print("Columns in training_jobs table:")
    print("=" * 50)
    for col in columns:
        print(f"  {col[1]:20s} - Type: {col[2]}")
    print("=" * 50)
    
    # Check if problem_type column exists
    column_names = [col[1] for col in columns]
    if 'problem_type' in column_names:
        print("\n✓ SUCCESS: 'problem_type' column exists!")
    else:
        print("\n✗ ERROR: 'problem_type' column is missing!")
    
    conn.close()
else:
    print("Database file not found. Please run the server once to create it.")
