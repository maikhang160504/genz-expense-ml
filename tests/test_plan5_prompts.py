"""
Unit tests cho KẾ HOẠCH 5 — Kiểm thử BotPromptsPage & 2 tầng NLU (forced_intent, rule_used, override_prompt).
"""
import sys
import os
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("src/api"))

from src.nlu.llm_intent_handler import run_llm_nlu_v2
from src.api.app.routers.nlu import test_prompt as api_test_prompt_endpoint


def test_run_llm_nlu_v2_forced_intent_record():
    """Kiểm chứng khi caller_context=addstory (force Record), Stage 1 được bỏ qua và trả về rule_used=record_rule."""
    fake_llm_response = '{"slots": {"item": "cà phê", "category": "Ăn uống", "amount": 25000}, "record_type": "Expense", "emotion": "dui_de", "response": "Mimo đã ghi nhận cà phê 25k nha"}'

    with patch("src.nlu.llm_intent_handler._call_llm", return_value=fake_llm_response) as mock_call:
        res = run_llm_nlu_v2(
            text="cà phê sáng 25k",
            context_metadata={"user_id": "test_user"},
            nlg_persona="dui_de",
            forced_intent="Record"
        )
        assert res["intent"] == "Record"
        assert res["rule_used"] == "record_rule"
        assert res["item"] == "cà phê"
        assert res["amount"] == 25000
        assert res["backend"] == "llm_v2"
        # Đảm bảo _call_llm được gọi
        mock_call.assert_called_once()


def test_run_llm_nlu_v2_override_prompt():
    """Kiểm chứng khi có override_prompt từ giao diện kiểm thử System Prompt."""
    fake_llm_response = '{"slots": {}, "emotion": "ngot_ngao", "response": "Mimo chào bạn nha"}'

    with patch("src.nlu.llm_intent_handler._call_llm", return_value=fake_llm_response) as mock_call:
        res = run_llm_nlu_v2(
            text="chào mimo",
            forced_intent="Chitchat",
            override_prompt="HỆ THỐNG GHI ĐÈ PROMPT CỦA MIMO"
        )
        assert res["intent"] == "Chitchat"
        assert res["rule_used"] == "chitchat_rule"
        # Kiểm tra arg system_prompt truyền vào _call_llm chính là override_prompt
        _, kwargs = mock_call.call_args
        assert kwargs["system_prompt"] == "HỆ THỐNG GHI ĐÈ PROMPT CỦA MIMO"


def test_router_test_prompt_addstory_force_record():
    """Kiểm chứng endpoint POST /test-prompt ánh xạ caller_context=addstory sang forced_intent=Record."""
    payload = {
        "text": "mua phở 50k",
        "caller_context": "addstory",
        "force_intent": "Auto",
        "persona": "dui_de"
    }

    with patch("src.nlu.llm_intent_handler.run_llm_nlu_v2") as mock_v2:
        mock_v2.return_value = {
            "intent": "Record",
            "rule_used": "record_rule",
            "amount": 50000,
            "category": "Ăn uống"
        }
        resp = api_test_prompt_endpoint(payload)
        assert resp["ok"] is True
        assert resp["result"]["rule_used"] == "record_rule"
        assert resp["result"]["intent"] == "Record"
        # Kiểm tra tham số gọi run_llm_nlu_v2 có forced_intent='Record'
        mock_v2.assert_called_once()
        _, kwargs = mock_v2.call_args
        assert kwargs["forced_intent"] == "Record"

