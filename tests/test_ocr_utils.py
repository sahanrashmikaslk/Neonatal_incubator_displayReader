from incubator_pipeline.ocr import clean_numeric


def test_clean_numeric_strips_noise():
    assert clean_numeric("SpO2 98%") == "98%"
    assert clean_numeric("36.3oC") == "36.3"
    assert clean_numeric("60- %") == "60%"


def test_clean_numeric_handles_empty():
    assert clean_numeric("") == ""
