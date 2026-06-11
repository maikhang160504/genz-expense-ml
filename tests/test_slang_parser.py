import pytest
from src.nlu.text import extract_amounts


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ăn trưa phở 45 cành", [45000]),
        ("cafe hết hai chục", [20000]),
        ("mua áo thun hết 3 loét", [300000]),
        ("mua giày củ rưỡi", [1500000]),
        ("mừng tuổi hai củ rưỡi", [2500000]),
        ("tiền thưởng nửa củ", [500000]),
        ("uống bia hết 2 lít", [200000]),
        ("mua điện thoại 15 quả", [15000000]),
        ("ăn buffet hết một mâm", [1000000]),
        ("nạp game hết năm chục cành", [50000]),
        ("mua xe hết mười củ", [10000000]),
        ("uống cf hết 25k", [25000]),
    ],
)
def test_slang_amount_extraction(text, expected):
    result = extract_amounts(text)
    assert result == expected
