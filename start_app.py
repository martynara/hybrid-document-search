import sys
import subprocess
import logging
import time
import signal
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cleanup_processes():
    try:
        from cleanup_processes import kill_python_processes, cleanup_qdrant_locks
        
        logger.info("Performing cleanup...")
        kill_python_processes()
        cleanup_qdrant_locks()
        time.sleep(1)
        
        return True
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")
        return False

def check_dependencies():
    required_modules = [
        'streamlit',
        'qdrant_client',
        'sentence_transformers',
        'openai',
        'spacy'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        logger.error("Please install dependencies:")
        logger.error("pip install streamlit qdrant-client sentence-transformers openai spacy")
        return False
    
    logger.info("All required dependencies are available")
    return True

def check_environment():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                logger.info("OpenAI API key configuration found")
            else:
                logger.warning("OpenAI API key not found in .env file")
    else:
        logger.warning(".env file not found - some features may not work without OpenAI API key")
    
    required_dirs = ["Data/Input/PDF", "Data/Output", "Data/VectorDB"]
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Environment check completed")
    return True

def start_streamlit(port=8501, cleanup_first=True):
    try:
        if cleanup_first:
            logger.info("Cleaning up before starting...")
            cleanup_processes()
        
        logger.info(f"Starting Streamlit application on port {port}...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run", "Home.py",
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(cmd)
        
        logger.info("=" * 60)
        logger.info(f"🚀 Application started successfully!")
        logger.info(f"📱 Local URL: http://localhost:{port}")
        logger.info(f"🌐 Network URL: http://<your-ip>:{port}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the application")
        
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Application stopped by user")
        try:
            process.terminate()
        except:
            pass
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Start HybridDocumentSearch application")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the application on")
    parser.add_argument("--skip-cleanup", action="store_true", help="Skip initial cleanup")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("🔍 HybridDocumentSearch - Startup Script")
    logger.info("=" * 60)
    
    if not args.skip_checks:
        logger.info("Step 1: Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        logger.info("Step 2: Checking environment...")
        if not check_environment():
            sys.exit(1)
    
    logger.info("Step 3: Starting application...")
    success = start_streamlit(port=args.port, cleanup_first=not args.skip_cleanup)
    
    if not success:
        logger.error("Failed to start application")
        sys.exit(1)

if __name__ == "__main__":
    main()