"""
Docling OCR FastAPI Server
==========================
Deploy on GPU VM. Receives PDF pages, returns structured text + tables.

Usage:
    uvicorn server:app --host 0.0.0.0 --port 8001
"""

import io
import base64
import tempfile
import shutil
import time
import json
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import local modules
from preprocessor import ImagePreprocessor, preprocess_pdf
from extractor import create_converter, EnhancedTextExtractor, TableExtractor
from postprocessor import TextSpacingCorrector

app = FastAPI(title="Docling OCR Server", version="1.0.0")


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class OCRRequest(BaseModel):
    """Request model for OCR processing."""
    pdf_base64: str  # Base64 encoded PDF bytes
    enable_preprocessing: bool = True
    enable_spacing_fix: bool = True
    enable_deskew: bool = True
    enable_denoising: bool = True
    target_dpi: int = 600
    ocr_lang: List[str] = ["en"]
    # Page range for batch processing (0-indexed)
    from_page: Optional[int] = None  # Inclusive
    to_page: Optional[int] = None    # Exclusive


class TableData(BaseModel):
    """Table data model."""
    index: int
    page_number: int
    rows: List[dict]


class PageData(BaseModel):
    """Page data model."""
    page_number: int
    text: str
    tables: List[TableData] = []


class OCRResponse(BaseModel):
    """Response model for OCR processing."""
    success: bool
    pages: List[PageData]
    total_pages: int
    total_tables: int
    processing_time: float
    error: Optional[str] = None


# =============================================================================
# CORE PROCESSING
# =============================================================================

def process_pdf(pdf_bytes: bytes, config: OCRRequest) -> dict:
    """
    Process PDF with Docling OCR.

    Args:
        pdf_bytes: Raw PDF bytes
        config: OCR configuration

    Returns:
        Dictionary with pages and metadata
    """
    from pypdf import PdfReader, PdfWriter

    start_time = time.time()
    tmpdir = tempfile.mkdtemp()
    page_offset = 0  # For correct page numbering in batch mode

    try:
        # Handle page range extraction for batch processing
        if config.from_page is not None or config.to_page is not None:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            total_in_pdf = len(reader.pages)
            start = config.from_page or 0
            end = min(config.to_page or total_in_pdf, total_in_pdf)

            print(f"[Batch] Extracting pages {start + 1}-{end} of {total_in_pdf}")

            writer = PdfWriter()
            for page_num in range(start, end):
                writer.add_page(reader.pages[page_num])

            # Save extracted pages to temp file
            batch_pdf_path = Path(tmpdir) / "batch.pdf"
            with open(batch_pdf_path, "wb") as f:
                writer.write(f)

            # Read the extracted batch
            with open(batch_pdf_path, "rb") as f:
                pdf_bytes = f.read()

            # Store offset for correct page numbering in response
            page_offset = start

        # Save PDF to temp file
        pdf_path = Path(tmpdir) / "input.pdf"
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        working_pdf = pdf_path

        # Optional preprocessing
        if config.enable_preprocessing:
            print(f"[Preprocessing] Deskew={config.enable_deskew}, Denoise={config.enable_denoising}, DPI={config.target_dpi}")
            preprocessor = ImagePreprocessor(
                enable_deskew=config.enable_deskew,
                enable_denoising=config.enable_denoising,
                target_dpi=config.target_dpi,
                debug=False
            )
            working_pdf = preprocess_pdf(pdf_path, preprocessor, Path(tmpdir))

        # Initialize corrector
        corrector = TextSpacingCorrector() if config.enable_spacing_fix else None

        # Create Docling converter and process
        print("[Docling] Creating converter...")
        converter = create_converter()
        print(f"[Docling] Processing {working_pdf}...")
        result = converter.convert(working_pdf)
        doc = result.document

        total_pages = len(doc.pages)
        print(f"[Docling] Extracted {total_pages} pages")

        # Extract tables with page numbers
        table_extractor = TableExtractor(corrector)
        extracted_tables = table_extractor.extract_tables(doc)
        print(f"[Tables] Extracted {len(extracted_tables)} tables")

        # Build table map by page
        table_page_map = {}  # page_num -> list of tables
        for table in extracted_tables:
            # Get page number from table provenance
            page_num = 1
            # Try to get from original table item
            for idx, tbl_item in enumerate(doc.tables, 1):
                if idx == table['index']:
                    if hasattr(tbl_item, 'prov') and tbl_item.prov:
                        for prov in tbl_item.prov:
                            if hasattr(prov, 'page_no'):
                                page_num = prov.page_no
                                break
                    break

            if page_num not in table_page_map:
                table_page_map[page_num] = []
            table_page_map[page_num].append({
                'index': table['index'],
                'page_number': page_num,
                'rows': table['dataframe'].to_dict('records')
            })

        # Extract text by page with embedded tables
        pages_result = []

        # Initialize pages
        page_content = {i: {'text_parts': [], 'table_indices': set()} for i in range(1, total_pages + 1)}

        current_table_idx = 1
        for item, level in doc.iterate_items():
            item_type = type(item).__name__

            # Get page number
            page_num = 1
            if hasattr(item, 'prov') and item.prov:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        page_num = max(1, min(prov.page_no, total_pages))
                        break

            if item_type == 'TableItem':
                # Mark where table appears in document order
                if page_num in page_content:
                    page_content[page_num]['text_parts'].append(f"__TABLE_{current_table_idx}__")
                current_table_idx += 1
                continue

            # Get text content
            text_content = None
            if hasattr(item, 'text') and item.text:
                text_content = item.text
            elif hasattr(item, 'export_to_text'):
                try:
                    text_content = item.export_to_text()
                except:
                    pass

            if text_content and text_content.strip():
                if corrector:
                    text_content = corrector.fix_spacing(text_content)

                if page_num in page_content:
                    page_content[page_num]['text_parts'].append(text_content)

        # Build final page data with embedded table JSON
        for page_num in range(1, total_pages + 1):
            parts = page_content.get(page_num, {}).get('text_parts', [])
            page_tables = table_page_map.get(page_num, [])

            # Replace table placeholders with JSON
            final_parts = []
            for part in parts:
                if part.startswith('__TABLE_') and part.endswith('__'):
                    # Extract table index and embed JSON
                    try:
                        tbl_num = int(part.replace('__TABLE_', '').replace('__', ''))
                        # Find matching table
                        for tbl in page_tables:
                            if tbl['index'] == tbl_num:
                                table_json = {"rows": tbl['rows']}
                                final_parts.append(f"\n<!-- TABLE {tbl_num} -->\n```json\n{json.dumps(table_json, indent=2, ensure_ascii=False)}\n```\n")
                                break
                    except:
                        pass
                else:
                    final_parts.append(part)

            page_text = '\n'.join(final_parts)

            pages_result.append({
                'page_number': page_num + page_offset,  # Add offset for batch mode
                'text': page_text,
                'tables': page_tables
            })

        processing_time = time.time() - start_time
        print(f"[Done] Processed in {processing_time:.2f}s")

        return {
            'success': True,
            'pages': pages_result,
            'total_pages': total_pages,
            'total_tables': len(extracted_tables),
            'processing_time': processing_time,
            'error': None
        }

    except Exception as e:
        import traceback
        error_msg = str(e) + '\n' + traceback.format_exc()
        print(f"[Error] {error_msg}")
        return {
            'success': False,
            'pages': [],
            'total_pages': 0,
            'total_tables': 0,
            'processing_time': time.time() - start_time,
            'error': error_msg
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "docling-ocr"}


@app.post("/v1/ocr/parse", response_model=OCRResponse)
async def parse_pdf(request: OCRRequest):
    """
    Parse PDF with Docling OCR.

    Receives base64-encoded PDF, returns structured page data with text and tables.
    """
    try:
        # Decode PDF
        pdf_bytes = base64.b64decode(request.pdf_base64)
        print(f"[Request] Received PDF: {len(pdf_bytes)} bytes")

        # Process
        result = process_pdf(pdf_bytes, request)

        return OCRResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/ocr/parse-pages")
async def parse_pdf_simple(request: OCRRequest):
    """
    Simplified endpoint returning just page texts (for RAGFlow integration).

    Returns list of {page_number, text} for direct chunking.
    """
    try:
        pdf_bytes = base64.b64decode(request.pdf_base64)
        print(f"[Request] Received PDF: {len(pdf_bytes)} bytes")

        result = process_pdf(pdf_bytes, request)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Processing failed'))

        # Return simplified format
        return {
            "pages": [
                {"page_number": p['page_number'], "text": p['text']}
                for p in result['pages']
            ],
            "total_pages": result['total_pages']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting Docling OCR Server on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
