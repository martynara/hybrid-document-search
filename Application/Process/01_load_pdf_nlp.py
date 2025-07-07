import logging
import os
from pathlib import Path

from Application.Pipelines.Load.PDF.LoadPDFPipeline import LoadPDFPipeline

# Configure logging to display in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    input_dir = "Data/Input/Pdf"
    output_dir = "Data/Input/Raw"

    Path(input_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return {"document": None}

    # Load PDF pipeline
    load_pdf_pipeline = LoadPDFPipeline(input_dir=input_dir, output_dir=output_dir)

    for i, pdf_file in enumerate(pdf_files):
        logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_file}")
        raw_data = load_pdf_pipeline.process_single_file(pdf_file)
        logger.info(f"Successfully processed {pdf_file}")

if __name__ == "__main__":
	main()