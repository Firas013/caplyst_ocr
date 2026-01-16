"""
PDF Extraction Module
======================
Handles core extraction of:
- Tables
- Text content
- Document structure
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
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend


class EnhancedTextExtractor:
    """
    Enhanced text extraction with better structure preservation.
    """

    def __init__(self, corrector=None):
        """
        Initialize text extractor.

        Args:
            corrector: Optional TextSpacingCorrector instance for fixing OCR errors
        """
        self.corrector = corrector

    def extract_from_document(self, doc) -> dict:
        """
        Extract text content from Docling document with enhanced structure.

        Args:
            doc: Docling document object

        Returns:
            dict with keys:
            - 'pages': dict mapping page numbers to list of text blocks
            - 'full_text': combined text with page markers
            - 'paragraphs': list of paragraph objects with metadata
            - 'sections': list of detected sections/headers
            - 'raw_text': plain text without formatting
        """
        import re

        pages_text = {}
        all_paragraphs = []
        sections = []

        # Initialize pages
        for page_no in range(1, len(doc.pages) + 1):
            pages_text[page_no] = []

        # Iterate through document items
        for item, level in doc.iterate_items():
            item_type = type(item).__name__

            # Skip table items (handled separately)
            if item_type == 'TableItem':
                continue

            text_content = None
            page_no = 1

            # Get text content
            if hasattr(item, 'text') and item.text:
                text_content = item.text
            elif hasattr(item, 'export_to_text'):
                try:
                    text_content = item.export_to_text()
                except:
                    pass

            # Get page number
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        page_no = prov.page_no
                        break

            if text_content and text_content.strip():
                # Apply spacing correction
                if self.corrector:
                    text_content = self.corrector.fix_spacing(text_content)

                # Determine element type
                element_type = self._classify_element(item_type, text_content, level)

                # Track sections/headers
                if element_type == 'heading':
                    sections.append({
                        'page': page_no,
                        'text': text_content,
                        'level': level
                    })

                paragraph_info = {
                    'page': page_no,
                    'type': element_type,
                    'text': text_content,
                    'level': level
                }

                all_paragraphs.append(paragraph_info)

                if page_no in pages_text:
                    pages_text[page_no].append({
                        'type': element_type,
                        'text': text_content
                    })

        # Build formatted full text
        full_text = self._build_formatted_text(pages_text)

        # Build raw text (no formatting)
        raw_text = self._build_raw_text(pages_text)

        return {
            'pages': pages_text,
            'full_text': full_text,
            'raw_text': raw_text,
            'paragraphs': all_paragraphs,
            'sections': sections,
            'page_count': len(doc.pages),
            'paragraph_count': len(all_paragraphs),
            'section_count': len(sections)
        }

    def _classify_element(self, item_type: str, text: str, level: int) -> str:
        """Classify the type of text element."""
        import re

        item_lower = item_type.lower()

        if 'header' in item_lower or 'title' in item_lower:
            return 'heading'
        elif 'list' in item_lower:
            return 'list_item'
        elif 'caption' in item_lower:
            return 'caption'
        elif 'footnote' in item_lower:
            return 'footnote'

        # Heuristic detection
        text_stripped = text.strip()

        # Check for numbered sections (e.g., "5- PROPERTY AND EQUIPMENT")
        if re.match(r'^\d+[-.)]\s*[A-Z]', text_stripped):
            return 'heading'

        # Check for all caps short text (likely heading)
        if len(text_stripped) < 100 and text_stripped.isupper():
            return 'heading'

        # Check for bullet points
        if text_stripped.startswith(('•', '-', '*', '○', '●')):
            return 'list_item'

        return 'paragraph'

    def _build_formatted_text(self, pages_text: dict) -> str:
        """Build formatted text with page markers and structure."""
        lines = []

        for page_no in sorted(pages_text.keys()):
            if not pages_text[page_no]:
                continue

            lines.append(f"\n{'='*60}")
            lines.append(f"PAGE {page_no}")
            lines.append('='*60)

            for block in pages_text[page_no]:
                text = block['text']
                element_type = block['type']

                if element_type == 'heading':
                    lines.append(f"\n## {text}\n")
                elif element_type == 'list_item':
                    lines.append(f"  • {text}")
                else:
                    lines.append(text)

        return '\n'.join(lines).strip()

    def _build_raw_text(self, pages_text: dict) -> str:
        """Build plain raw text."""
        lines = []
        for page_no in sorted(pages_text.keys()):
            for block in pages_text[page_no]:
                lines.append(block['text'])
        return '\n'.join(lines).strip()


class TableExtractor:
    """Handles table extraction from documents."""

    def __init__(self, corrector=None):
        """
        Initialize table extractor.

        Args:
            corrector: Optional TextSpacingCorrector instance for fixing OCR errors
        """
        self.corrector = corrector

    def extract_tables(self, doc) -> list:
        """
        Extract all tables from a document.

        Args:
            doc: Docling document object

        Returns:
            List of dicts with 'index' and 'dataframe' keys
        """
        print("\n[Extracting Tables...]")
        extracted_tables = []

        for idx, table in enumerate(doc.tables, 1):
            try:
                df = table.export_to_dataframe(doc=doc)
                if df.empty:
                    print(f"  [Skip] Table {idx} is empty")
                    continue

                if self.corrector:
                    df = self.corrector.fix_dataframe(df)

                print(f"  [OK] Table {idx}: {df.shape[0]} rows x {df.shape[1]} cols")
                extracted_tables.append({
                    'index': idx,
                    'dataframe': df
                })
            except Exception as e:
                print(f"  [Error] Table {idx}: {e}")

        return extracted_tables


def create_converter():
    """
    Create Docling converter with OCR enabled.

    Returns:
        DocumentConverter instance
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE

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
        print(f"[Info] Using local PP-OCRv3 models")
    else:
        ocr_options = RapidOcrOptions(lang=["en"], force_full_page_ocr=True)
        print("[Info] Using default RapidOCR configuration")

    pipeline_options.ocr_options = ocr_options

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
    Extract tables and text from PDF document.

    Args:
        pdf_path: Path to PDF file
        corrector: Optional TextSpacingCorrector instance

    Returns:
        Tuple of (doc, extracted_tables, text_data)
    """
    print("\nInitializing Docling converter...")
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