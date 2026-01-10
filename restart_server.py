"""
Complete Server Restart Script
This will stop the server, fix the database, and restart it.
"""
import subprocess
import os
import time

print("=" * 60)
print("COMPLETE SERVER RESTART")
print("=" * 60)

# Step 1: Stop all Python processes running the server
print("\n1. Stopping existing Python processes...")
try:
    # Kill uvicorn/python processes (be careful!)
    result = subprocess.run(
        ["taskkill", "/F", "/FI", "IMAGENAME eq python*.exe", "/FI", "WINDOWTITLE eq *uvicorn*"],
        capture_output=True,
        text=True
    )
    print("   Attempted to stop server processes")
    time.sleep(2)
except Exception as e:
    print(f"   Note: {e}")

# Step 2: Fix the database
print("\n2. Fixing database...")
try:
    db_path = "users.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"   ✓ Deleted old database")
    
    # Create new database
    from main import Base, engine
    Base.metadata.create_all(bind=engine)
    print(f"   ✓ Created new database with correct schema")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    print("\n   MANUAL FIX NEEDED:")
    print("   1. Close any terminals running Python")
    print("   2. Delete users.db manually")
    print("   3. Restart the server")
    exit(1)

# Step 3: Instructions to restart
print("\n3. Database is ready! Now start the server:")
print("\n   Run this command in a new terminal:")
print("   ┌─────────────────────────────────────────────────────────┐")
print("   │  uvicorn main:app --reload --port 8081                  │")
print("   └─────────────────────────────────────────────────────────┘")
print("\n   Or use:")
print("   ┌─────────────────────────────────────────────────────────┐")
print("   │  python main.py                                         │")
print("   └─────────────────────────────────────────────────────────┘")
print("\n" + "=" * 60)
print("READY! Start the server with one of the commands above.")
print("=" * 60)
