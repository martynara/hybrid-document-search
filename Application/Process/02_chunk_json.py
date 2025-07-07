import logging
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from Application.Domain.RawData import RawData
from Application.Pipelines.Chunking.ChunkingPipeline import ChunkingPipeline

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():

    raw_input_dir = "Data/Input/Raw"
    output_dir = "Data/Output"

    # Create absolute paths
    input_dir = os.path.join(project_root, raw_input_dir)
    output_dir = os.path.join(project_root, output_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all Raw files in the input directory
    raw_json_files = [f for f in os.listdir(raw_input_dir) if f.lower().endswith('.json')]
    
    if not raw_json_files:
        logger.warning(f"No JSON files found in {raw_input_dir}")
        return {"document": None}

    # Create chunking pipeline
    chunking_pipeline = ChunkingPipeline(input_dir=raw_input_dir, output_dir=output_dir)

    for i, raw_json_file in enumerate(raw_json_files):
        logger.info(f"Processing file {i+1}/{len(raw_json_files)}: {raw_json_file}")

        # Load RawData from the input file for processing
        raw_json_file_path = os.path.join(input_dir, raw_json_file)
        raw_data = RawData.load_from_file(raw_json_file_path)

        # Chunking
        document = chunking_pipeline.process_raw_data(raw_data)

        # Save document
        base_name, file_ext = os.path.splitext(raw_json_file)
        output_file_path = os.path.join(output_dir, f"{base_name}.json")
        document.save_to_file(output_file_path)

        # logger.info(f"Successfully processed {json_file} to {output_file_path}")

if __name__ == "__main__":
    main()
