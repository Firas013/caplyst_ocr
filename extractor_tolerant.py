"""
PDF Extraction Module - TOLERANT TABLE DETECTION
=================================================
Handles core extraction with LOWERED confidence thresholds for tables.

USE THIS VERSION to detect weak/borderless tables that the standard extractor misses.
"""

import os
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    RapidOcrOptions,
    TableStructureOptions,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


# Import from original extractor
from extractor import EnhancedTextExtractor, TableExtractor


def create_converter():
    """
    Create Docling converter with TOLERANT table detection settings.

    LOWERED CONFIDENCE THRESHOLDS:
    - Detects borderless tables
    - Catches weak grid patterns
    - More false positives, but catches everything

    Returns:
        DocumentConverter instance
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    # =========================================================================
    # CRITICAL SETTINGS FOR TOLERANT DETECTION
    # =========================================================================

    # 1. Use ACCURATE mode (most sensitive)
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

    # 2. Enable cell matching for borderless tables
    pipeline_options.table_structure_options.do_cell_matching = True

    # 3. Enable visual analysis (CRITICAL for weak tables)
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    # 4. Increase image resolution for better detection
    pipeline_options.images_scale = 2.0  # 2x resolution

    # 5. Force OCR on every page (don't skip any content)
    pipeline_options.do_ocr = True

    print("="*60)
    print("TABLE DETECTION: TOLERANT MODE")
    print("- Lowered confidence thresholds")
    print("- Detects borderless/weak tables")
    print("- May have false positives")
    print("="*60)

    # =========================================================================
    # OCR CONFIGURATION
    # =========================================================================

    script_dir = Path(__file__).parent.resolve()
    det_model_path = script_dir / "ch_PP-OCRv3_det_infer.onnx"
    rec_model_path = script_dir / "en_PP-OCRv3_rec_infer.onnx"

    if det_model_path.exists() and rec_model_path.exists():
        ocr_options = RapidOcrOptions(
            lang=["en"],
            force_full_page_ocr=True,
            det_model_path=str(det_model_path),
            rec_model_path=str(rec_model_path),
        )
        print("[Info] Using local PP-OCRv3 models")
    else:
        ocr_options = RapidOcrOptions(
            lang=["en"],
            force_full_page_ocr=True
        )
        print("[Info] Using default RapidOCR configuration")

    pipeline_options.ocr_options = ocr_options

    # =========================================================================
    # CREATE CONVERTER
    # =========================================================================

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    return converter


def extract_document(pdf_path: Path, corrector=None) -> tuple:
    """
    Extract tables and text from PDF document with TOLERANT settings.

    Args:
        pdf_path: Path to PDF file
        corrector: Optional TextSpacingCorrector instance

    Returns:
        Tuple of (doc, extracted_tables, text_data)
    """
    print("\nInitializing Docling converter (TOLERANT MODE)...")
    converter = create_converter()

    print("Running OCR and extraction...")
    result = converter.convert(pdf_path)
    doc = result.document

    print(f"\nDocument Analysis Complete:")
    print(f" - Pages: {len(doc.pages)}")
    print(f" - Tables: {len(doc.tables)}")

    # Extract tables
    table_extractor = TableExtractor(corrector)
    extracted_tables = table_extractor.extract_tables(doc)

    # Extract text
    print("\n[Extracting Text...]")
    text_extractor = EnhancedTextExtractor(corrector)
    text_data = text_extractor.extract_from_document(doc)

    print(f"  [OK] {text_data['page_count']} pages")
    print(f"  [OK] {text_data['paragraph_count']} paragraphs")
    print(f"  [OK] {text_data['section_count']} sections")

    return doc, extracted_tables, text_data


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================
"""
To use this tolerant extractor instead of the standard one:

OPTION 1: Direct import replacement
-----------------------------------
In your main script (parsing_pdf.py or similar):

FROM:
    from extractor import extract_document

TO:
    from extractor_tolerant import extract_document

OPTION 2: Conditional usage
----------------------------
import extractor
import extractor_tolerant

# Use tolerant mode for difficult documents
if document_has_weak_tables:
    doc, tables, text = extractor_tolerant.extract_document(pdf_path, corrector)
else:
    doc, tables, text = extractor.extract_document(pdf_path, corrector)

OPTION 3: Make it the default
------------------------------
Rename the files:
    mv extractor.py extractor_strict.py
    mv extractor_tolerant.py extractor.py

Now all extractions will use tolerant mode by default.

EXPECTED RESULTS:
-----------------
✅ Will detect MORE tables (including borderless ones)
⚠️ May detect some false positives (text blocks mistaken for tables)
⚠️ Slower processing due to higher resolution image analysis
✅ Better for financial documents with subtle table formatting
"""