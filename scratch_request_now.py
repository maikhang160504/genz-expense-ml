import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

def main():
    url = "http://127.0.0.1:8000/api/v1/nlu/infer"
    data = {
        "text": "mua trà sữa hết 45k",
        "profile": {
            "budget_total": 5000000,
            "budget_remain": 1200000
        },
        "run_llm": True,
        "nlg_persona": "dan_doi"
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    print("Sending request to NLU service...")
    sys.stdout.flush()
    
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            print("Response received successfully:")
            print(json.dumps(res_json, indent=2, ensure_ascii=False))
            sys.stdout.flush()
    except Exception as e:
        print(f"Request failed: {e}")
        if hasattr(e, "read"):
            print(e.read().decode("utf-8"))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
