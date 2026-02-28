# tests/test_integration_pdf.py

from app import load_pdf_text_from_path

def test_real_pdf_loads():
    text = load_pdf_text_from_path("philippine_history.pdf")
    
    assert len(text) > 10000
    assert "Philippine" in text  # sanity check