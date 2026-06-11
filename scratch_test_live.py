import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def test_nlu(text, persona):
    url = "http://localhost:8000/api/v1/nlu/infer"
    data = {
        "text": text,
        "profile": {
            "budget_total": 5000000,
            "budget_remain": 1200000
        },
        "run_llm": True,
        "nlg_persona": persona
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    print(f"\n======================================")
    print(f"TESTING TEXT: '{text}' WITH PERSONA: '{persona}'")
    print(f"======================================")
    
    try:
        with urllib.request.urlopen(req) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            
            print("\n--- [NLU PIPELINE OUTPUT] ---")
            print(f"Intent: {res_json.get('intent')}")
            print(f"Category: {res_json.get('category')}")
            print(f"Amount: {res_json.get('amount')}")
            print(f"Record Type: {res_json.get('record_type')}")
            print(f"Backend: {res_json.get('backend')}")
            print(f"Latency: {res_json.get('latency_ms')} ms")
            print(f"Mimo Emotion/Mascot Mood: {res_json.get('mascot_mood')}")
            print(f"NLG Response: {res_json.get('nlg_response')}")
            
            nlg_prompt = res_json.get("nlg_prompt")
            if nlg_prompt:
                print("\n--- [NLG PROMPT SENT TO GEMINI] ---")
                print(">>> SYSTEM PROMPT:")
                print(nlg_prompt.get("system"))
                print("\n>>> USER PROMPT:")
                print(nlg_prompt.get("user"))
            
            gemini_json = res_json.get("gemini_json")
            if gemini_json:
                print("\n--- [GEMINI RAW JSON RESPONSE] ---")
                print(json.dumps(gemini_json, indent=2, ensure_ascii=False))
                
            raw_response = res_json.get("gemini_response")
            if raw_response:
                print("\n--- [GEMINI RAW STRING RESPONSE] ---")
                print(raw_response)
                
    except Exception as e:
        print(f"Request failed: {e}")
        if hasattr(e, "read"):
            print(e.read().decode("utf-8"))

if __name__ == "__main__":
    test_nlu("mua trà sữa hết 45k", "dan_doi")
    test_nlu("lương về 15tr", "hai_huoc")
