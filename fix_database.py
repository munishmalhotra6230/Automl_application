"""
Database Fix Script
This script will delete the old database and create a new one with the problem_type column.
Run this AFTER stopping your FastAPI server.
"""
import os
import sqlite3
import time

db_path = "users.db"

print("=" * 60)
print("DATABASE FIX SCRIPT")
print("=" * 60)

# Step 1: Check if database exists
if os.path.exists(db_path):
    print(f"\n1. Found existing database: {db_path}")
    
    # Try to delete it
    try:
        os.remove(db_path)
        print(f"   ✓ Successfully deleted old database")
    except PermissionError:
        print(f"   ✗ ERROR: Database is locked (server is still running)")
        print(f"\n   PLEASE STOP THE SERVER FIRST!")
        print(f"   Press Ctrl+C in the terminal where uvicorn is running")
        print(f"   Then run this script again.\n")
        exit(1)
    except Exception as e:
        print(f"   ✗ ERROR: Could not delete database: {e}")
        exit(1)
else:
    print(f"\n1. No existing database found (this is OK)")

# Step 2: Import the models and create new database
print(f"\n2. Creating new database with updated schema...")
try:
    from main import Base, engine, TrainingJob
    Base.metadata.create_all(bind=engine)
    print(f"   ✓ Database created successfully")
except Exception as e:
    print(f"   ✗ ERROR: Failed to create database: {e}")
    exit(1)

# Step 3: Verify the schema
print(f"\n3. Verifying database schema...")
try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(training_jobs)")
    columns = cursor.fetchall()
    
    print(f"\n   Columns in training_jobs table:")
    for col in columns:
        print(f"      - {col[1]:20s} ({col[2]})")
    
    # Check if problem_type exists
    column_names = [col[1] for col in columns]
    if 'problem_type' in column_names:
        print(f"\n   ✓ SUCCESS! 'problem_type' column is present")
        print(f"\n" + "=" * 60)
        print(f"DATABASE FIXED! You can now restart your server.")
        print(f"=" * 60)
    else:
        print(f"\n   ✗ ERROR: 'problem_type' column is still missing!")
        print(f"   Please check your main.py file.")
    
    conn.close()
except Exception as e:
    print(f"   ✗ ERROR: Failed to verify schema: {e}")
    exit(1)
