"""
Region Detection Module for Preprocessing
==========================================
Add these methods to the ImagePreprocessor class in preprocessor.py
"""

import cv2
import numpy as np


def _detect_and_enhance_regions(self, image: np.ndarray, page_num: int = 0) -> np.ndarray:
    """
    Detect and enhance text/table regions using contour detection.

    This identifies distinct content blocks (text paragraphs, tables, images)
    and enhances them individually for better OCR accuracy.

    Args:
        image: Input image as numpy array
        page_num: Page number (for debug output)

    Returns:
        Enhanced image with region-specific processing
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological operations to connect nearby text
    kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    # Detect horizontal and vertical lines (for tables)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_horizontal)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_vertical)

    # Combine to detect table regions
    table_mask = cv2.bitwise_or(horizontal, vertical)

    # Dilate to group text blocks
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilated = cv2.dilate(binary, kernel_dilate, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create output image
    result = image.copy()

    # Debug: draw contours if debug mode is on
    if self.debug and self.debug_dir:
        debug_img = image.copy()

    # Process each region
    regions = []
    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out very small regions (noise)
        if w < 50 or h < 20:
            continue

        # Extract region
        region = image[y:y+h, x:x+w]

        # Determine if it's likely a table (has grid lines)
        table_region = table_mask[y:y+h, x:x+w]
        is_table = np.sum(table_region) / (w * h) > 0.05  # 5% threshold

        # Apply region-specific enhancement
        if is_table:
            # Table region: enhance grid lines and text
            enhanced_region = self._enhance_table_region(region)
            color = (0, 255, 0)  # Green for tables
            region_type = "TABLE"
        else:
            # Text region: enhance text clarity
            enhanced_region = self._enhance_text_region(region)
            color = (255, 0, 0)  # Blue for text
            region_type = "TEXT"

        # Replace region in result
        result[y:y+h, x:x+w] = enhanced_region

        regions.append({
            'type': region_type,
            'bbox': (x, y, w, h),
            'area': w * h
        })

        # Draw bounding box in debug mode
        if self.debug and self.debug_dir:
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(debug_img, f"{region_type}_{idx}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save debug image with detected regions
    if self.debug and self.debug_dir:
        cv2.imwrite(str(self.debug_dir / f"page_{page_num}_regions_detected.png"), debug_img)
        print(f"  [Debug] Detected {len(regions)} regions on page {page_num}")
        for r in regions:
            print(f"    - {r['type']}: {r['bbox']} (area: {r['area']})")

    return result


def _enhance_table_region(self, region: np.ndarray) -> np.ndarray:
    """Enhance table regions for better grid and text detection."""
    # Sharpen for better line detection
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(region, -1, kernel_sharpen)

    # Increase contrast
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced


def _enhance_text_region(self, region: np.ndarray) -> np.ndarray:
    """Enhance text regions for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Apply adaptive threshold for better text clarity
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Convert back to BGR
    enhanced = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return enhanced


# =============================================================================
# INTEGRATION INSTRUCTIONS
# =============================================================================
"""
To integrate into preprocessor.py:

1. Add this to the process_image() method, after denoising:

    if self.enable_segmentation:
        image = self._detect_and_enhance_regions(image, page_num)

2. Add the three methods above to the ImagePreprocessor class:
   - _detect_and_enhance_regions()
   - _enhance_table_region()
   - _enhance_text_region()

3. When you enable debug mode and segmentation, you'll get:
   - Bounding boxes around detected regions (green=tables, blue=text)
   - Debug image saved as "page_N_regions_detected.png"
   - Console output listing all detected regions with coordinates

This will improve extraction accuracy by:
- Identifying table vs text regions BEFORE extraction
- Applying specialized enhancement to each region type
- Tables get sharpened for better grid detection
- Text gets adaptive thresholding for clearer characters
"""