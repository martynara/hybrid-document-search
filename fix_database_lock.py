#!/usr/bin/env python3
"""
Quick fix script for Qdrant database lock issues.
Run this whenever you get "Storage folder Data/VectorDB is already accessed" error.
"""

import os
import subprocess
import glob
import time

print("🔧 Fixing Qdrant Database Lock Issues...")
print("=" * 50)

# Step 1: Kill any Python processes
try:
    print("1. Killing Python processes...")
    if os.name == 'nt':  # Windows
        subprocess.run(["taskkill", "/f", "/im", "python.exe"], 
                      capture_output=True, text=True)
    else:  # Unix-like
        subprocess.run(["pkill", "-f", "python"], 
                      capture_output=True, text=True)
    print("   ✅ Python processes killed")
except:
    print("   ⚠️  Could not kill processes (might be none running)")

# Step 2: Remove lock files
print("2. Removing lock files...")
lock_files = glob.glob("Data/VectorDB/**/*.lock", recursive=True)
lock_files.extend(glob.glob("Data/VectorDB/.lock"))

if lock_files:
    for lock_file in lock_files:
        try:
            os.remove(lock_file)
            print(f"   ✅ Removed: {lock_file}")
        except:
            print(f"   ❌ Could not remove: {lock_file}")
else:
    print("   ✅ No lock files found")

# Step 3: Wait a moment
print("3. Waiting for cleanup...")
time.sleep(2)

print("=" * 50)
print("🎉 Database lock fix completed!")
print("You can now start your application:")
print("   streamlit run home.py")
print("   OR")
print("   python start_app.py --skip-cleanup") 