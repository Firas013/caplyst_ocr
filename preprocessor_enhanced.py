"""
PDF Image Preprocessing Module - ENHANCED VERSION
==================================================
Handles image preprocessing including:
- Deskewing
- Denoising
- Enhancement
- TABLE BORDER ENHANCEMENT (NEW!)
- PDF to image conversion
"""

from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


class ImagePreprocessor:
    """Image pre-processing for better OCR results with table enhancement."""

    def __init__(self, enable_segmentation=False, enable_deskew=True,
                 enable_denoising=True, enable_table_enhancement=True,
                 target_dpi=600, debug=False):
        """
        Initialize the image preprocessor.

        Args:
            enable_segmentation: Whether to enable segmentation
            enable_deskew: Whether to enable deskewing
            enable_denoising: Whether to enable denoising
            enable_table_enhancement: Whether to enhance table borders (NEW!)
            target_dpi: Target DPI for image conversion
            debug: Whether to save debug images
        """
        self.enable_segmentation = enable_segmentation
        self.enable_deskew = enable_deskew
        self.enable_denoising = enable_denoising
        self.enable_table_enhancement = enable_table_enhancement
        self.target_dpi = target_dpi
        self.debug = debug
        self.debug_dir = None

    def set_debug_dir(self, debug_dir: Path):
        """Set the debug directory for saving intermediate images."""
        self.debug_dir = Path(debug_dir)
        if self.debug:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

    def process_image(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Process a single image with all enabled preprocessing steps.

        Args:
            image: Input image as numpy array
            page_num: Page number (for debug naming)

        Returns:
            Processed image as numpy array
        """
        if self.debug and self.debug_dir:
            cv2.imwrite(str(self.debug_dir / f"page_{page_num}_0_original.png"), image)

        if self.enable_deskew:
            image = self._deskew_image(image)
            if self.debug and self.debug_dir:
                cv2.imwrite(str(self.debug_dir / f"page_{page_num}_1_deskewed.png"), image)

        if self.enable_denoising:
            image = self._denoise_and_enhance(image)
            if self.debug and self.debug_dir:
                cv2.imwrite(str(self.debug_dir / f"page_{page_num}_2_denoised.png"), image)

        # NEW: Table border enhancement
        if self.enable_table_enhancement:
            image = self._enhance_table_borders(image, page_num)
            if self.debug and self.debug_dir:
                cv2.imwrite(str(self.debug_dir / f"page_{page_num}_3_table_enhanced.png"), image)

        return image

    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew an image by detecting dominant angles and rotating.

        Args:
            image: Input image as numpy array

        Returns:
            Deskewed image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100,
                                minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) == 0:
            return image

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 45:
                    angles.append(angle)

        if not angles:
            return image

        median_angle = np.median(angles)

        if 0.5 < abs(median_angle) < 15:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return image

    def _denoise_and_enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise and enhance image using CLAHE and bilateral filtering.

        Args:
            image: Input image as numpy array

        Returns:
            Enhanced image
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised

    def _enhance_table_borders(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
        """
        Enhance table borders to help detection of borderless tables.

        This method detects table-like structures (aligned rows/columns) and
        subtly emphasizes their boundaries without modifying the text content.

        Args:
            image: Input image as numpy array
            page_num: Page number (for debug output)

        Returns:
            Image with enhanced table borders
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detect horizontal lines (table rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # Detect vertical lines (table columns)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Combine horizontal and vertical lines
        table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

        # Dilate slightly to connect nearby lines
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_structure = cv2.dilate(table_structure, dilate_kernel, iterations=1)

        # Find contours of table regions
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create result image
        result = image.copy()

        # Draw subtle borders around detected tables
        table_count = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter: must be reasonably large to be a table
            if w > 100 and h > 50:
                # Draw very subtle gray border (won't interfere with OCR but helps detection)
                cv2.rectangle(result, (x-2, y-2), (x+w+2, y+h+2), (180, 180, 180), 1)

                # Enhance the detected table region slightly
                table_region = result[y:y+h, x:x+w]

                # Sharpen just this region for better text detection
                kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                           [-0.5,  5.5, -0.5],
                                           [-0.5, -0.5, -0.5]])
                table_region = cv2.filter2D(table_region, -1, kernel_sharpen)
                result[y:y+h, x:x+w] = table_region

                table_count += 1

        if self.debug:
            print(f"  [Table Enhancement] Page {page_num}: Detected {table_count} potential tables")

        # If debug, save the table structure mask
        if self.debug and self.debug_dir:
            cv2.imwrite(str(self.debug_dir / f"page_{page_num}_table_mask.png"), table_structure)

        return result


def preprocess_pdf(pdf_path: Path, preprocessor: ImagePreprocessor, output_dir: Path) -> Path:
    """
    Preprocess PDF images for better OCR.

    Args:
        pdf_path: Path to input PDF
        preprocessor: ImagePreprocessor instance
        output_dir: Directory to save preprocessed PDF

    Returns:
        Path to preprocessed PDF
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.set_debug_dir(output_dir / "debug_images")

    print("\n[Pre-processing] Converting PDF to images...")
    images = convert_from_path(pdf_path, dpi=preprocessor.target_dpi)
    print(f"[Pre-processing] Converted {len(images)} pages")

    processed_images = []
    for idx, pil_image in enumerate(images):
        print(f"[Pre-processing] Processing page {idx + 1}/{len(images)}...")
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        processed = preprocessor.process_image(cv_image, page_num=idx + 1)
        processed_pil = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
        processed_images.append(processed_pil)

    output_pdf = output_dir / f"{pdf_path.stem}_preprocessed.pdf"

    if processed_images:
        rgb_images = [img.convert('RGB') if img.mode != 'RGB' else img for img in processed_images]
        rgb_images[0].save(
            str(output_pdf),
            "PDF",
            save_all=True,
            append_images=rgb_images[1:] if len(rgb_images) > 1 else [],
            resolution=preprocessor.target_dpi
        )
        print(f"[Pre-processing] Saved preprocessed PDF: {output_pdf}")

    return output_pdf