#!/usr/bin/env python3
"""
Utility script to clean up running Python/Streamlit processes 
that might be holding Qdrant database locks.

Run this script before starting the application if you encounter
database lock errors.
"""

import os
import sys
import subprocess
import logging
import platform

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def kill_python_processes():
    """Kill running Python processes that might be holding database locks."""
    system = platform.system()
    
    try:
        if system == "Windows":
            # Windows approach
            logger.info("Killing Python processes on Windows...")
            result = subprocess.run(
                ["taskkill", "/f", "/im", "python.exe"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                logger.info("Successfully killed Python processes")
            else:
                logger.warning(f"taskkill returned code {result.returncode}: {result.stderr}")
                
            # Also try to kill streamlit specifically
            result = subprocess.run(
                ["taskkill", "/f", "/im", "streamlit.exe"],
                capture_output=True,
                text=True
            )
            
        else:
            # Unix-like systems (Linux, macOS)
            logger.info("Killing Python processes on Unix-like system...")
            
            # Find Python processes
            result = subprocess.run(
                ["pgrep", "-f", "python"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                logger.info(f"Found Python processes: {pids}")
                
                # Kill each process
                for pid in pids:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        logger.info(f"Killed process {pid}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"Failed to kill process {pid}")
            else:
                logger.info("No Python processes found")
                
            # Also try to kill streamlit processes
            result = subprocess.run(
                ["pkill", "-f", "streamlit"],
                capture_output=True,
                text=True
            )
            
    except FileNotFoundError:
        logger.error("Required command not found. Please manually kill Python processes.")
        if system == "Windows":
            logger.info("Try: taskkill /f /im python.exe")
        else:
            logger.info("Try: pkill -f python")
    except Exception as e:
        logger.error(f"Error killing processes: {str(e)}")

def cleanup_qdrant_locks():
    """Clean up any remaining Qdrant database locks."""
    db_path = "Data/VectorDB"
    
    if os.path.exists(db_path):
        logger.info(f"Checking Qdrant database directory: {db_path}")
        
        # Look for any lock files or temporary files
        for root, dirs, files in os.walk(db_path):
            for file in files:
                if file.endswith(('.lock', '.tmp', '.temp')):
                    lock_file = os.path.join(root, file)
                    try:
                        os.remove(lock_file)
                        logger.info(f"Removed lock file: {lock_file}")
                    except Exception as e:
                        logger.warning(f"Could not remove {lock_file}: {str(e)}")
    else:
        logger.info("Qdrant database directory does not exist yet")

def main():
    """Main cleanup function."""
    logger.info("=" * 60)
    logger.info("HybridDocumentSearch - Process Cleanup Utility")
    logger.info("=" * 60)
    
    # Step 1: Kill Python processes
    logger.info("Step 1: Killing Python processes...")
    kill_python_processes()
    
    # Step 2: Clean up Qdrant locks
    logger.info("Step 2: Cleaning up Qdrant database locks...")
    cleanup_qdrant_locks()
    
    # Step 3: Wait a moment
    import time
    logger.info("Step 3: Waiting for cleanup to complete...")
    time.sleep(2)
    
    logger.info("=" * 60)
    logger.info("Cleanup completed! You can now start the application.")
    logger.info("Run: streamlit run Home.py")
    logger.info("=" * 60)

if __name__ == "__main__":
    main() 