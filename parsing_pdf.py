"""
Enhanced PDF Table & Text Extraction
====================================
Improved text extraction with better handling of:
- Financial documents
- Multi-column layouts
- Headers and sections
- Currency symbols (like SAR ﷼)

This is the main entry point that orchestrates the modular components:
- preprocessor: Image preprocessing (deskew, denoise, enhance)
- extractor: Core extraction (tables, text, structure)
- postprocessor: Text correction and output formatting
"""

import sys
import time
from pathlib import Path

# Import modular components
from preprocessor import ImagePreprocessor, preprocess_pdf
from extractor import extract_document
from postprocessor import TextSpacingCorrector, OutputSaver


# =============================================================================
# MAIN EXTRACTION FUNCTION
# =============================================================================

def extract_all(pdf_path: str, output_dir: str = "./output",
                apply_spacing_fix: bool = True,
                preprocess: bool = True) -> dict:
    """
    Extract tables AND text from PDF.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        apply_spacing_fix: Whether to apply OCR text spacing corrections
        preprocess: Whether to preprocess the PDF images

    Returns:
        dict with:
        - 'tables': list of extracted table DataFrames
        - 'text': enhanced text extraction results
        - 'time': processing time in seconds
        - 'output_files': dict of saved file paths
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n{'='*60}")
    print(f"EXTRACTING: {pdf_path.name}")
    print(f"Spacing Fix: {'ON' if apply_spacing_fix else 'OFF'}")
    print(f"Preprocessing: {'ON' if preprocess else 'OFF'}")
    print('='*60)

    start_time = time.time()

    # Initialize corrector
    corrector = TextSpacingCorrector() if apply_spacing_fix else None

    # Optional preprocessing
    working_pdf = pdf_path
    if preprocess:
        preprocessor = ImagePreprocessor(
            enable_segmentation=False,  # Can cause issues with some PDFs
            enable_deskew=True,
            enable_denoising=True,
            debug=False
        )
        working_pdf = preprocess_pdf(pdf_path, preprocessor, output_path)

    # Extract document
    doc, extracted_tables, text_data = extract_document(working_pdf, corrector)

    extraction_time = time.time() - start_time

    print(f"\nDocument Analysis Complete:")
    print(f" - Pages: {len(doc.pages)}")
    print(f" - Tables: {len(extracted_tables)}")
    print(f" - Time: {extraction_time:.2f}s")

    # Save outputs
    saver = OutputSaver(output_path, pdf_path.stem, corrector)
    saver.save_all(doc, extracted_tables, text_data, extraction_time, pdf_path.name)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"Output directory: {output_path}")
    print('='*60)

    return {
        'tables': extracted_tables,
        'text': text_data,
        'time': extraction_time,
        'table_count': len(extracted_tables),
        'output_files': saver.get_output_files()
    }


# =============================================================================
# SIMPLE TEXT-ONLY EXTRACTION (Alternative method)
# =============================================================================

def extract_text_only(pdf_path: str, output_path: str = None) -> str:
    """
    Simple function to extract just the text from a PDF.

    Args:
        pdf_path: Path to PDF file
        output_path: Optional path to save text file

    Returns:
        Extracted text as string
    """
    from extractor import create_converter, EnhancedTextExtractor

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Extracting text from: {pdf_path.name}")

    converter = create_converter()
    result = converter.convert(pdf_path)
    doc = result.document

    corrector = TextSpacingCorrector()
    extractor = EnhancedTextExtractor(corrector)
    text_data = extractor.extract_from_document(doc)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_data['full_text'])
        print(f"Saved to: {output_path}")

    return text_data['full_text']


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python parsing_pdf.py <pdf_file> [options]")
        print("\nOptions:")
        print("  --no-fix         Disable spacing correction")
        print("  --no-preprocess  Disable image preprocessing")
        print("  --text-only      Extract only text (faster)")
        print("  --output DIR     Output directory (default: ./output)")
        sys.exit(1)

    pdf_input = sys.argv[1]
    no_fix = "--no-fix" in sys.argv
    no_preprocess = "--no-preprocess" in sys.argv
    text_only = "--text-only" in sys.argv

    output_dir = "./output"
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        if idx + 1 < len(sys.argv):
            output_dir = sys.argv[idx + 1]

    print("="*60)
    print("ENHANCED PDF EXTRACTOR")
    print("="*60)

    try:
        if text_only:
            text = extract_text_only(pdf_input, f"{output_dir}/extracted_text.txt")
            print(f"\n--- EXTRACTED TEXT PREVIEW ---")
            print(text[:2000] + "..." if len(text) > 2000 else text)
        else:
            results = extract_all(
                pdf_input,
                output_dir=output_dir,
                apply_spacing_fix=not no_fix,
                preprocess=not no_preprocess
            )

            print(f"\nSummary:")
            print(f"  Tables extracted: {results['table_count']}")
            print(f"  Text paragraphs: {results['text']['paragraph_count']}")
            print(f"  Processing time: {results['time']:.2f}s")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
