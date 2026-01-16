"""
Post-Processing Module
======================
Handles post-processing of extracted data including:
- Text spacing correction
- Data cleaning
- Output formatting
- File saving
"""

import re
import json
from pathlib import Path
import pandas as pd

try:
    import wordninja
    WORDNINJA_AVAILABLE = True
except ImportError:
    WORDNINJA_AVAILABLE = False
    print("[Warning] wordninja not installed. Install with: pip install wordninja")


class TextSpacingCorrector:
    """Fixes merged words in OCR output."""

    def __init__(self):
        pass

    def fix_spacing(self, text: str) -> str:
        """
        Fix spacing in text by splitting merged words.

        Args:
            text: Input text string

        Returns:
            Text with corrected spacing
        """
        if not text or not isinstance(text, str):
            return text
        if len(text) < 5:
            return text

        if ' ' in text and len(text.split()) > 1:
            words = text.split()
            fixed_words = [self._fix_word(w) for w in words]
            return ' '.join(fixed_words)

        return self._fix_word(text)

    def _fix_word(self, word: str) -> str:
        """Fix a single word."""
        if not word or len(word) < 5:
            return word

        if self._has_case_transitions(word):
            result = self._split_camel_case(word)
            if result != word:
                return result

        if WORDNINJA_AVAILABLE and len(word) > 10:
            result = self._apply_wordninja(word)
            if result != word:
                return result

        return word

    def _has_case_transitions(self, text: str) -> bool:
        """Check if text has lowercase to uppercase transitions."""
        for i in range(1, len(text)):
            if text[i-1].islower() and text[i].isupper():
                return True
        return False

    def _split_camel_case(self, text: str) -> str:
        """Split camelCase or PascalCase text."""
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', result)
        return result

    def _apply_wordninja(self, text: str) -> str:
        """Apply wordninja library to split merged words."""
        if not WORDNINJA_AVAILABLE:
            return text
        if not text.replace('-', '').replace('_', '').isalpha():
            return text
        try:
            words = wordninja.split(text.lower())
            if len(words) > 1:
                return ' '.join(words)
        except:
            pass
        return text

    def fix_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply spacing correction to all text in a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with corrected text
        """
        df_fixed = df.copy()
        for col in df_fixed.columns:
            if df_fixed[col].dtype == 'object':
                df_fixed[col] = df_fixed[col].apply(
                    lambda x: self.fix_spacing(x) if isinstance(x, str) else x
                )
        df_fixed.columns = [self.fix_spacing(str(col)) for col in df_fixed.columns]
        return df_fixed


class OutputSaver:
    """Handles saving extracted data to various formats."""

    def __init__(self, output_dir: Path, base_name: str, corrector=None):
        """
        Initialize output saver.

        Args:
            output_dir: Directory to save output files
            base_name: Base name for output files
            corrector: Optional TextSpacingCorrector instance
        """
        self.output_dir = Path(output_dir)
        self.base_name = base_name
        self.corrector = corrector
        self.output_files = {}

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_tables(self, extracted_tables: list):
        """
        Save tables to Excel and CSV formats.

        Args:
            extracted_tables: List of table dictionaries with 'index' and 'dataframe'
        """
        if not extracted_tables:
            return

        # Save to Excel
        excel_path = self.output_dir / f"{self.base_name}_tables.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for item in extracted_tables:
                sheet_name = f"Table_{item['index']}"[:31]
                item['dataframe'].to_excel(writer, sheet_name=sheet_name, index=False)
        self.output_files['excel'] = str(excel_path)
        print(f"  → {excel_path.name}")

        # Save individual CSVs
        for item in extracted_tables:
            csv_path = self.output_dir / f"{self.base_name}_table_{item['index']}.csv"
            item['dataframe'].to_csv(csv_path, index=False)

    def save_text(self, text_data: dict):
        """
        Save text data in various formats.

        Args:
            text_data: Dictionary containing text extraction results
        """
        # Save formatted text
        txt_path = self.output_dir / f"{self.base_name}_text.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text_data['full_text'])
        self.output_files['text'] = str(txt_path)
        print(f"  → {txt_path.name}")

        # Save raw text (no formatting)
        raw_txt_path = self.output_dir / f"{self.base_name}_text_raw.txt"
        with open(raw_txt_path, 'w', encoding='utf-8') as f:
            f.write(text_data['raw_text'])
        self.output_files['raw_text'] = str(raw_txt_path)
        print(f"  → {raw_txt_path.name}")

    def save_json(self, text_data: dict, extracted_tables: list, extraction_time: float, source_file: str):
        """
        Save structured JSON output.

        Args:
            text_data: Dictionary containing text extraction results
            extracted_tables: List of extracted tables
            extraction_time: Processing time in seconds
            source_file: Source PDF filename
        """
        json_path = self.output_dir / f"{self.base_name}_extracted.json"
        json_output = {
            'source_file': source_file,
            'extraction_time': extraction_time,
            'page_count': text_data['page_count'],
            'table_count': len(extracted_tables),
            'paragraph_count': text_data['paragraph_count'],
            'sections': text_data['sections'],
            'paragraphs': text_data['paragraphs'],
            'tables': [
                {
                    'index': t['index'],
                    'rows': t['dataframe'].shape[0],
                    'cols': t['dataframe'].shape[1],
                    'data': t['dataframe'].to_dict('records')
                }
                for t in extracted_tables
            ]
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)
        self.output_files['json'] = str(json_path)
        print(f"  → {json_path.name}")

    def save_markdown(self, doc, extracted_tables: list):
        """
        Save markdown output with tables embedded as pure JSON in document order.

        Args:
            doc: Docling document object
            extracted_tables: List of extracted tables with 'index' and 'dataframe'
        """
        # Create a mapping of table indices to their data
        table_map = {t['index']: t['dataframe'] for t in extracted_tables}

        md_lines = []
        current_table_index = 1

        # Iterate through document items in order
        for item, level in doc.iterate_items():
            item_type = type(item).__name__

            # Handle tables - embed as pure JSON
            if item_type == 'TableItem':
                if current_table_index in table_map:
                    df = table_map[current_table_index]

                    # Convert table to pure JSON (just the rows)
                    table_json = {
                        "rows": df.to_dict('records')
                    }

                    # Add table marker and JSON
                    md_lines.append(f"\n<!-- TABLE {current_table_index} -->\n")
                    md_lines.append("```json")
                    md_lines.append(json.dumps(table_json, indent=2, ensure_ascii=False))
                    md_lines.append("```\n")

                current_table_index += 1
                continue

            # Handle text content
            text_content = None
            if hasattr(item, 'text') and item.text:
                text_content = item.text
            elif hasattr(item, 'export_to_text'):
                try:
                    text_content = item.export_to_text()
                except:
                    pass

            if text_content and text_content.strip():
                # Apply spacing correction if available
                if self.corrector:
                    text_content = self.corrector.fix_spacing(text_content)

                # Format based on item type
                if 'SectionHeaderItem' in item_type or 'Title' in item_type:
                    md_lines.append(f"\n## {text_content}\n")
                elif 'ListItem' in item_type:
                    md_lines.append(f"- {text_content}")
                else:
                    md_lines.append(text_content)

        md_content = '\n'.join(md_lines)

        md_path = self.output_dir / f"{self.base_name}_content.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        self.output_files['markdown'] = str(md_path)
        print(f"  → {md_path.name}")

    def save_all(self, doc, extracted_tables: list, text_data: dict, extraction_time: float, source_file: str):
        """
        Save all output formats.

        Args:
            doc: Docling document object
            extracted_tables: List of extracted tables
            text_data: Dictionary containing text extraction results
            extraction_time: Processing time in seconds
            source_file: Source PDF filename
        """
        print("\n[Saving outputs...]")
        # Only save markdown with embedded JSON tables - no separate files needed
        self.save_markdown(doc, extracted_tables)

    def get_output_files(self) -> dict:
        """
        Get dictionary of saved output files.

        Returns:
            Dictionary mapping format names to file paths
        """
        return self.output_files