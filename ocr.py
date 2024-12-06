
python
Copy code
import cv2
import pytesseract
from pytesseract import Output
from pdf2image import convert_from_path
import numpy as np

# Set path to Tesseract executable if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def convert_pdf_to_images(pdf_path, dpi=300):
    """
    Convert PDF to images using pdf2image.
    """
    return convert_from_path(pdf_path, dpi=dpi)

def preprocess_image(image):
    """
    Preprocess the image for better OCR results.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh

def perform_ocr(image, psm_mode=6):
    """
    Perform OCR on an image using Tesseract with layout analysis.
    """
    # Convert image to RGB for Tesseract compatibility
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tesseract configuration
    custom_config = f'--oem 3 --psm {psm_mode}'

    # Perform OCR and extract detailed layout data
    ocr_result = pytesseract.image_to_data(rgb_image, output_type=Output.DICT, config=custom_config)
    return ocr_result

def extract_text_and_structure(ocr_data):
    """
    Process OCR results to organize text and detect sections like tables and narratives.
    """
    extracted_text = []
    table_data = []
    current_section = []

    for i, text in enumerate(ocr_data['text']):
        if text.strip():  # Ignore empty text
            # Collect text line by line
            x, y, w, h = (ocr_data['left'][i], ocr_data['top'][i],
                          ocr_data['width'][i], ocr_data['height'][i])
            conf = int(ocr_data['conf'][i])

            if conf > 50:  # Only consider text with a confidence score above 50
                current_section.append((x, y, w, h, text))
                extracted_text.append(text)

                # Check if this line resembles a table row (basic heuristic)
                if w > 0.8 * max(ocr_data['width']):  # Assuming wide content is a table row
                    table_data.append(text)

    return {
        "extracted_text": "\n".join(extracted_text),
        "table_data": table_data
    }

def save_to_text_file(output_path, text, tables):
    """
    Save the extracted text and tables to a text file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Extracted Text:\n")
        f.write(text)
        f.write("\n\nTabular Data:\n")
        f.writelines("\n".join(tables))

def process_pdf(pdf_path, output_txt_path, dpi=300, psm_mode=6):
    """
    Full pipeline to extract text and tables from a PDF and save as a text file.
    """
    # Convert PDF pages to images
    images = convert_pdf_to_images(pdf_path, dpi=dpi)

    all_text = []
    all_tables = []

    for page_number, image in enumerate(images):
        # Convert PIL image to OpenCV format
        open_cv_image = np.array(image)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # Preprocess the image
        processed_image = preprocess_image(open_cv_image)

        # Perform OCR
        ocr_data = perform_ocr(processed_image, psm_mode=psm_mode)

        # Extract text and structure
        result = extract_text_and_structure(ocr_data)

        # Append results
        all_text.append(f"--- Page {page_number + 1} ---\n" + result["extracted_text"])
        all_tables.extend(result["table_data"])

    # Save results to text file
    save_to_text_file(output_txt_path, "\n".join(all_text), all_tables)

# Example usage
pdf_path = "path/to/your/input.pdf"
output_txt_path = "path/to/your/output.txt"

process_pdf(pdf_path, output_txt_path)
print(f"Text extracted and saved to {output_txt_path}")
