#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated NLU Scenario Verification Tool.
Tests Qwen LLM slot classification, prompt adjustments (like trà sữa -> Food),
first-pass empty Action response, second-pass action commentary,
emotion synchronization, and personalization overrides.
"""
import os
import sys
import json
import argparse
import httpx

DEFAULT_BASE_URL = "https://maikhang160504--expense-ocr-nlu-fastapi-app.modal.run"

TEST_SCENARIOS = [
    # 1. Record: category Food mapping
    {
        "name": "Food category map rule (tea)",
        "payload": {
            "text": "mua trà sữa hết 45k",
            "runLlm": True,
            "nlgPersona": "hai_huoc"
        },
        "assertions": {
            "intent": "Record",
            "category": "Food",
            "record_type": "Expense",
            "amount": 45000,
            "item_contains": "trà sữa"
        }
    },
    # 2. Record: category Food mapping (coffee)
    {
        "name": "Food category map rule (coffee)",
        "payload": {
            "text": "tôi đã mua 1 ly cà phê sữa 20k",
            "runLlm": True,
            "nlgPersona": "hai_huoc"
        },
        "assertions": {
            "intent": "Record",
            "category": "Food",
            "record_type": "Expense",
            "amount": 20000,
            "item_contains": "cà phê"
        }
    },
    # 3. Record: Shopping classification
    {
        "name": "Shopping category classification",
        "payload": {
            "text": "chi 100k mua áo thun",
            "runLlm": True,
            "nlgPersona": "hai_huoc"
        },
        "assertions": {
            "intent": "Record",
            "category": "Shopping",
            "record_type": "Expense",
            "amount": 100000,
            "item_contains": "áo thun"
        }
    },
    # 4. Record: Income classification
    {
        "name": "Income record classification",
        "payload": {
            "text": "lương tháng này được 15 triệu",
            "runLlm": True,
            "nlgPersona": "hai_huoc"
        },
        "assertions": {
            "intent": "Record",
            "category": "Salary",
            "record_type": "Income",
            "amount": 15000000,
            "item_contains": "lương"
        }
    },
    # 5. Action: First Pass (Response must be empty)
    {
        "name": "Action: First Pass (empty response)",
        "payload": {
            "text": "tổng chi tiêu tuần này",
            "runLlm": True,
            "nlgPersona": "nghiem_tuc"
        },
        "assertions": {
            "intent": "Action",
            "action_type": "REPORT_GENERAL",
            "response_empty": True
        }
    },
    # 6. Action: Second Pass Commentary (response must contain facts commentary)
    {
        "name": "Action: Second Pass Commentary",
        "payload": {
            "text": "tổng chi tiêu tuần này",
            "runLlm": True,
            "nlgPersona": "hai_huoc",
            "profile": {
                "action_facts": {
                    "total_spent": 1200000,
                    "by_category": {
                        "Food": 600000,
                        "Shopping": 400000
                    }
                }
            }
        },
        "assertions": {
            "intent": "Action",
            "action_type": "REPORT_GENERAL",
            "response_not_empty": True,
            "response_contains_any": ["1.200.000", "ăn uống", "mua sắm", "600", "400"]
        }
    },
    # 7. Chitchat
    {
        "name": "Chitchat conversation",
        "payload": {
            "text": "hello mimo",
            "runLlm": True,
            "nlgPersona": "hai_huoc"
        },
        "assertions": {
            "intent": "Chitchat",
            "response_not_empty": True
        }
    },
    # 8. Personalization override
    {
        "name": "Personalization override priority",
        "payload": {
            "text": "uống trà",
            "runLlm": True,
            "user_corrections": [
                {
                    "text": "uống trà",
                    "category_code": "Social",
                    "record_type": "Expense",
                    "intent": "Record"
                }
            ]
        },
        "assertions": {
            "intent": "Record",
            "category": "Social",
            "record_type": "Expense",
            "backend_contains": "user_"
        }
    }
]

def run_scenarios(base_url: str):
    print(f"🚀 Starting automated NLU scenario tests against: {base_url}\n")
    client = httpx.Client(timeout=120.0)
    
    passed_count = 0
    failed_count = 0
    
    for idx, sc in enumerate(TEST_SCENARIOS, 1):
        print(f"Test case {idx}: {sc['name']}")
        url = f"{base_url}/api/v1/nlu/infer"
        
        try:
            res = client.post(url, json=sc["payload"])
            if res.status_code != 200:
                print(f"  ❌ FAILED: Status code {res.status_code}")
                print(f"  Response: {res.text}\n")
                failed_count += 1
                continue
                
            data = res.json()
            errors = []
            
            # Run assertions
            assertions = sc["assertions"]
            if "intent" in assertions and data.get("intent") != assertions["intent"]:
                errors.append(f"intent: expected '{assertions['intent']}', got '{data.get('intent')}'")
                
            if "category" in assertions and data.get("category") != assertions["category"]:
                errors.append(f"category: expected '{assertions['category']}', got '{data.get('category')}'")
                
            if "record_type" in assertions and data.get("record_type") != assertions["record_type"]:
                errors.append(f"record_type: expected '{assertions['record_type']}', got '{data.get('record_type')}'")
                
            if "amount" in assertions and data.get("amount") != assertions["amount"]:
                errors.append(f"amount: expected {assertions['amount']}, got {data.get('amount')}")
                
            if "action_type" in assertions and data.get("action_type") != assertions["action_type"]:
                errors.append(f"action_type: expected '{assertions['action_type']}', got '{data.get('action_type')}'")
                
            if "item_contains" in assertions:
                item = str(data.get("item") or "").lower()
                expected = assertions["item_contains"].lower()
                if expected not in item:
                    errors.append(f"item: expected to contain '{expected}', got '{item}'")
                    
            if assertions.get("response_empty") and str(data.get("nlg_response") or "").strip() != "":
                errors.append(f"nlg_response: expected empty string, got '{data.get('nlg_response')}'")
                
            if assertions.get("response_not_empty") and str(data.get("nlg_response") or "").strip() == "":
                errors.append("nlg_response: expected non-empty string, got empty/null")
                
            if "response_contains_any" in assertions:
                resp = str(data.get("nlg_response") or "").lower()
                candidates = assertions["response_contains_any"]
                found = any(c.lower() in resp for c in candidates)
                if not found:
                    errors.append(f"nlg_response: expected to contain any of {candidates}, got '{resp}'")
                    
            if "backend_contains" in assertions:
                be = str(data.get("backend") or "").lower()
                expected = assertions["backend_contains"].lower()
                if expected not in be:
                    errors.append(f"backend: expected to contain '{expected}', got '{be}'")
                    
            # Check emotion list safety
            emo = data.get("mimo_emotion")
            if emo:
                # Sanity check: no CJK characters in response
                resp_text = str(data.get("nlg_response") or "")
                cjk_chars = [c for c in resp_text if ord(c) >= 0x3000 and ord(c) <= 0x9FFF]
                if cjk_chars:
                    errors.append(f"nlg_response: leaked CJK/Chinese characters: {''.join(cjk_chars)}")
            
            if errors:
                print(f"  ❌ FAILED: {', '.join(errors)}")
                print(f"  Response JSON: {json.dumps(data, ensure_ascii=False, indent=2)}\n")
                failed_count += 1
            else:
                print(f"  ✅ PASSED (Latency: {data.get('latency_ms', '?')}ms, Emotion: {data.get('mimo_emotion', 'None')})\n")
                passed_count += 1
                
        except Exception as e:
            print(f"  ❌ CRITICAL ERROR calling FastAPI: {e}\n")
            failed_count += 1

    print("==================================================")
    print(f"📊 Test Suite Finished: {passed_count} Passed, {failed_count} Failed.")
    print("==================================================")
    
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="Base API url to run NLU scenarios against")
    args = parser.parse_args()
    run_scenarios(args.url)
