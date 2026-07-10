# Save as tests/test_nlu_llm.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import json
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "text_nlu"))

from src.nlu.pipeline import run_nlu

MOCK_RESPONSE = {
    "intent": "Record",
    "action_type": None,
    "slots": {
        "category": "Food",
        "amount": 25000,
        "note": "cà phê sữa đá"
    },
    "emotion": "happy",
    "response": "Mimo ghi bún bò ngon lành nha!"
}

def test_llm_nlu_pipeline():
    print("=== Running NLU Unified LLM Pipeline Unit Test ===")
    with patch("src.nlu.llm_intent_handler._call_llm", return_value=json.dumps(MOCK_RESPONSE)):
        with patch("src.nlu.models.get_inference_backend", return_value="llm"):
            res = run_nlu(
                "mua cà phê sữa đá 25k",
                None, None, None, None, None, None, None
            )
            
            print("Parsed output response:")
            print(json.dumps(res, indent=2, ensure_ascii=False))
            
            # Assert NLU properties match mocked output
            assert res["intent"] == "Record"
            assert res["category"] == "Food"
            assert res["amount"] == 25000
            assert res["mimo_emotion"] == "happy"
            assert res["nlg_response"] == "Mimo ghi bún bò ngon lành nha!"
            assert res["backend"] == "llm_unified"
            print("Unit test 'test_llm_nlu_pipeline' PASSED!")

if __name__ == "__main__":
    test_llm_nlu_pipeline()
