"""
Test script cho KẾ HOẠCH 2 & 3:
- Test endpoint /api/v1/nlu/infer với backend llm (unified) hiện tại
- Bao gồm: 1 Record, tất cả action_type (11 loại), 1 Chitchat, 1 AddStory shortcut
- Kiểm tra cấu trúc response và tính đúng đắn
"""
import json
import time
import sys
import os
import urllib.request
import urllib.error

USE_LOCAL_CLIENT = os.environ.get("NLU_TEST_MODE", "remote").lower() == "local"
_local_client = None

def get_local_client():
    global _local_client
    if _local_client is None:
        from fastapi.testclient import TestClient
        from src.api.app.main import create_app
        _local_client = TestClient(create_app())
    return _local_client

BASE_URL = "https://maikhang160504--expense-ocr-nlu-fastapi-app.modal.run"

TEST_CASES = [
    # === RECORD ===
    {
        "name": "Record — ghi chép chi tiêu có số tiền",
        "payload": {"text": "ăn phở với bạn hết 45k", "run_llm": True, "nlg_persona": "dui_de"},
        "expect_intent": "Record",
    },
    {
        "name": "Record — ghi chép chi tiêu KHÔNG có số tiền",
        "payload": {"text": "vừa mua cái áo mới", "run_llm": True},
        "expect_intent": "Record",
    },

    # === ACTION — 11 action_type ===
    {
        "name": "Action — REPORT_GENERAL",
        "payload": {"text": "tháng này tôi tiêu hết bao nhiêu rồi?", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "REPORT_GENERAL",
    },
    {
        "name": "Action — REPORT_COMPARE",
        "payload": {"text": "tháng này so với tháng trước chi tiêu của tôi thay đổi thế nào?", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "REPORT_COMPARE",
    },
    {
        "name": "Action — SET_LIMIT",
        "payload": {"text": "đặt hạn mức ăn uống tháng này 3 triệu", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SET_LIMIT",
    },
    {
        "name": "Action — SET_GOAL (saving_personal)",
        "payload": {"text": "tạo mục tiêu tiết kiệm 10 triệu mua xe trong 6 tháng", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SET_GOAL",
    },
    {
        "name": "Action — ADD_GOAL",
        "payload": {"text": "nạp thêm 500 nghìn vào quỹ tiết kiệm mua xe", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "ADD_GOAL",
    },
    {
        "name": "Action — SET_TONE",
        "payload": {"text": "đổi giọng điệu sang nghiêm túc đi Mimo", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SET_TONE",
    },
    {
        "name": "Action — SET_ALERT",
        "payload": {"text": "bật cảnh báo chi tiêu cho tôi", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SET_ALERT",
    },
    {
        "name": "Action — SYSTEM_SETTING (dark mode)",
        "payload": {"text": "chuyển giao diện sang chế độ tối", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SYSTEM_SETTING",
    },
    {
        "name": "Action — SEARCH_RECORD",
        "payload": {"text": "liệt kê tất cả giao dịch tuần này", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SEARCH_RECORD",
    },
    {
        "name": "Action — SUGGEST_BUDGET",
        "payload": {"text": "gợi ý ngân sách ăn uống phù hợp cho tôi tháng này", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SUGGEST_BUDGET",
    },
    {
        "name": "Action — SET_USERNAME",
        "payload": {"text": "gọi tôi là Khang nhé Mimo", "run_llm": True},
        "expect_intent": "Action",
        "expect_action_type": "SET_USERNAME",
    },

    # === CHITCHAT ===
    {
        "name": "Chitchat — nói chuyện phiếm",
        "payload": {"text": "hôm nay trời đẹp quá Mimo ơi", "run_llm": True},
        "expect_intent": "Chitchat",
    },

    # === ADDSTORY SHORTCUT (KẾ HOẠCH 3) ===
    # Dùng backend llm hiện tại, test field caller_context
    {
        "name": "caller_context=addstory — text chitchat phải ra Record",
        "payload": {"text": "hôm nay trời đẹp quá", "run_llm": True, "caller_context": "addstory"},
        "expect_intent": "Record",  # Force Record vì addstory
        "note": "Test caller_context field mới — field này được nhận bởi server nhưng logic addstory chỉ active khi backend=llm_v2",
    },
]


def call_infer(payload: dict) -> dict | None:
    if USE_LOCAL_CLIENT:
        try:
            resp = get_local_client().post("/api/v1/nlu/infer", json=payload)
            if resp.status_code == 200:
                return resp.json()
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except Exception as ex:
            print(f"  Error local client: {ex}")
            return None

    url = f"{BASE_URL}/api/v1/nlu/infer"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body[:200]}")
        return None
    except Exception as ex:
        print(f"  Error: {ex}")
        return None


def switch_backend(backend: str = "llm_v2") -> bool:
    if USE_LOCAL_CLIENT:
        try:
            resp = get_local_client().post("/api/v1/nlu/inference-backend", json={"backend": backend})
            if resp.status_code == 200:
                res = resp.json()
                print(f"[+] Switched local NLU inference backend to: {res.get('backend', backend)}")
                return True
            print(f"[!] Warning switch backend HTTP {resp.status_code}")
            return False
        except Exception as e:
            print(f"[!] Error switch backend local client: {e}")
            return False

    url = f"{BASE_URL}/api/v1/nlu/inference-backend"
    data = json.dumps({"backend": backend}).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            print(f"[+] Switched NLU inference backend to: {res.get('backend', backend)}")
            return True
    except Exception as e:
        print(f"[!] Warning: Could not switch backend to {backend} via API: {e}")
        return False


def run_tests():
    results = []
    passed = 0
    failed = 0

    print("=" * 65)
    print("TEST KẾ HOẠCH 2 & 3 — NLU Infer với llm_rules.json (llm_v2)")
    print("=" * 65)

    switch_backend("llm_v2")

    for tc in TEST_CASES:
        name = tc["name"]
        payload = tc["payload"]
        expect_intent = tc.get("expect_intent")
        expect_action = tc.get("expect_action_type")
        note = tc.get("note", "")

        print(f"\n── {name} ──")
        print(f"   Input: {payload['text'][:70]}")
        if note:
            print(f"   NOTE: {note}")

        t0 = time.monotonic()
        resp = call_infer(payload)
        latency = int((time.monotonic() - t0) * 1000)

        if resp is None:
            print(f"   STATUS: FAIL (không nhận được response)")
            failed += 1
            results.append({"name": name, "pass": False, "error": "no_response", "latency_ms": latency})
            continue

        got_intent = resp.get("intent")
        got_action = resp.get("action_type")
        got_response = resp.get("nlg_response") or resp.get("response") or ""
        got_backend = resp.get("backend", "?")
        got_emotion = resp.get("mimo_emotion") or resp.get("mascot_mood") or "?"
        got_suggested = resp.get("suggested_actions")

        # Kiểm tra
        intent_ok = (expect_intent is None) or (got_intent == expect_intent)
        action_ok = (expect_action is None) or (got_action == expect_action)

        status = "PASS" if (intent_ok and action_ok) else "FAIL"
        if intent_ok and action_ok:
            passed += 1
        else:
            failed += 1

        print(f"   STATUS: {status} | Intent: {got_intent} | ActionType: {got_action} | Backend: {got_backend}")
        print(f"   Emotion: {got_emotion} | Latency: {latency}ms")
        print(f"   Response: {got_response[:100]}")
        if got_suggested:
            print(f"   Suggested: {got_suggested}")
        if not intent_ok:
            print(f"   [!] Intent mong đợi: {expect_intent}, thực tế: {got_intent}")
        if not action_ok:
            print(f"   [!] ActionType mong đợi: {expect_action}, thực tế: {got_action}")

        results.append({
            "name": name,
            "pass": intent_ok and action_ok,
            "intent": got_intent,
            "action_type": got_action,
            "backend": got_backend,
            "emotion": got_emotion,
            "nlg_response": got_response[:150],
            "suggested_actions": got_suggested,
            "latency_ms": latency,
        })
        sys.stdout.flush()
        with open("test_kh23_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 65)
    print(f"KẾT QUẢ: {passed} PASS / {failed} FAIL / {len(TEST_CASES) - passed - failed} WARN")
    print("=" * 65)

    with open("test_kh23_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Đã lưu kết quả vào test_kh23_results.json")
    return results


if __name__ == "__main__":
    run_tests()
