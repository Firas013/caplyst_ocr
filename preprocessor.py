"""
PDF Image Preprocessing Module
================================
Handles image preprocessing including:
- Deskewing
- Denoising
- Enhancement
- PDF to image conversion
"""

from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image


class ImagePreprocessor:
    """Image pre-processing for better OCR results."""

    def __init__(self, enable_segmentation=False, enable_deskew=True,
                 enable_denoising=False, target_dpi=600, debug=False):
        """
        Initialize the image preprocessor.

        Args:
            enable_segmentation: Whether to enable segmentation
            enable_deskew: Whether to enable deskewing
            enable_denoising: Whether to enable denoising
            target_dpi: Target DPI for image conversion
            debug: Whether to save debug images
        """
        self.enable_segmentation = enable_segmentation
        self.enable_deskew = enable_deskew
        self.enable_denoising = enable_denoising
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

        if self.enable_denoising:
            image = self._denoise_and_enhance(image)

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