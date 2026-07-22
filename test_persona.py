# -*- coding: utf-8 -*-
"""
Test script: Kiem tra LLM NLU persona injection sau khi cap nhat.
Chay tu thu muc: d:\Luan-Van\Project\expense-ocr-nlu
Lenh: python test_persona.py
"""
import sys, os, json, datetime

# Load .env TRUOC KHI import bat ky module nao
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

sys.path.insert(0, ".")
os.environ["RUN_LLM"] = "1"


# Nap prompts_config truc tiep de kiem tra key mapping
with open("src/prompts/prompts.json", encoding="utf-8") as f:
    prompts_config = json.load(f)

emotions_keys = list(prompts_config.get("emotions", {}).keys())
mood_keys = list(prompts_config.get("mood_personas", {}).keys())
print(f"[DEBUG] emotions keys: {emotions_keys}")
print(f"[DEBUG] mood_personas keys: {mood_keys}")
print()

# --- Import ham LLM ---
from src.nlu.llm_intent_handler import run_llm_nlu

TEST_CASES = [
    {
        "label": "DUI_DE - Chi tieu binh thuong",
        "text": "an trua bun bo 45k",
        "persona": "dui_de",
        "context": {
            "username": "Khang",
            "wallet_health": "tot",
            "time_of_day": "trua",
            "days_to_payday": 10
        }
    },
    {
        "label": "DAN_DOI - Chi tieu lon",
        "text": "mua iPhone 25 trieu",
        "persona": "dan_doi",
        "context": {
            "username": "Khang",
            "wallet_health": "can",
            "time_of_day": "toi",
            "days_to_payday": 3
        }
    },
    {
        "label": "KHO_TINH - Chi cafe",
        "text": "mua cafe Starbucks 120k",
        "persona": "kho_tinh",
        "context": {
            "username": "Khang",
            "wallet_health": "binh thuong",
            "time_of_day": "sang"
        }
    },
    {
        "label": "NGOT_NGAO - Tieu nhieu",
        "text": "di spa 800k",
        "persona": "ngot_ngao",
        "context": {
            "username": "Khang",
            "wallet_health": "vua phai",
            "time_of_day": "chieu"
        }
    },
    {
        "label": "DUI_DE - Quan he NGUOI_YEU (chi cho bo)",
        "text": "mua qua tang nguoi yeu 300k",
        "persona": "dui_de",
        "context": {
            "username": "Khang",
            "wallet_health": "tot",
        }
    },
    {
        "label": "DAN_DOI - Quan he CHA_ME (bieu me)",
        "text": "bieu me 500k",
        "persona": "dan_doi",
        "context": {
            "username": "Khang",
            "wallet_health": "tot"
        }
    },
    {
        "label": "KHO_TINH - Chitchat hoi tham",
        "text": "hom nay toi tieu nhieu qua, co cach nao tiet kiem khong?",
        "persona": "kho_tinh",
        "context": {
            "username": "Khang",
            "wallet_health": "can",
            "days_to_payday": 5
        }
    },
    {
        "label": "NGOT_NGAO - Thu nhap",
        "text": "nhan luong thang nay 8 trieu",
        "persona": "ngot_ngao",
        "context": {
            "username": "Khang",
            "wallet_health": "tot",
            "time_of_day": "sang"
        }
    },
]

results = []
for tc in TEST_CASES:
    print(f"Testing: {tc['label']}...")
    try:
        result = run_llm_nlu(
            text=tc["text"],
            context_metadata=tc.get("context"),
            run_llm=True,
            nlg_persona=tc["persona"]
        )
        response = result.get("response", "(Khong co phan hoi)")
        emotion = result.get("emotion", "?")
        intent = result.get("intent", "?")
        results.append({
            "label": tc["label"],
            "input": tc["text"],
            "persona": tc["persona"],
            "intent": intent,
            "emotion": emotion,
            "response": response,
            "ok": True
        })
        print(f"  -> [{emotion}] {response}\n")
    except Exception as e:
        results.append({
            "label": tc["label"],
            "input": tc["text"],
            "persona": tc["persona"],
            "error": str(e),
            "ok": False
        })
        print(f"  ERROR: {e}\n")

# Xuat file Markdown
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
md = f"""# Ket qua Test LLM Persona Prompt
**Thoi gian:** {now}

---

## Tong quan

| # | Test Case | Persona | Intent | Emotion | Ket qua |
|---|-----------|---------|--------|---------|---------|
"""

for i, r in enumerate(results, 1):
    if r["ok"]:
        md += f"| {i} | {r['label']} | `{r['persona']}` | {r['intent']} | {r['emotion']} | OK |\n"
    else:
        md += f"| {i} | {r['label']} | `{r['persona']}` | - | - | LOI |\n"

md += "\n---\n\n## Chi tiet tung Test Case\n\n"

for i, r in enumerate(results, 1):
    md += f"### {i}. {r['label']}\n"
    md += f"- **Input:** `{r['input']}`\n"
    md += f"- **Persona:** `{r['persona']}`\n"
    if r["ok"]:
        md += f"- **Intent phat hien:** `{r['intent']}`\n"
        md += f"- **Emotion:** `{r['emotion']}`\n"
        md += f"- **Phan hoi AI:**\n\n> {r['response']}\n\n"
    else:
        md += f"- **LOI:** `{r['error']}`\n\n"

md += "---\n\n## Nhan xet & Danh gia\n\n"
md += "- [ ] DUI_DE co thuc su vui ve, nang luong cao khong?\n"
md += "- [ ] DAN_DOI co thuc su can nhang, xeo xat khong?\n"
md += "- [ ] KHO_TINH co thuc su nghiem khac, ky luat khong?\n"
md += "- [ ] NGOT_NGAO co thuc su ngot ngao, chua lanh khong?\n"
md += "- [ ] Quan he CHA_ME co doi sang giong am ap khong?\n"
md += "- [ ] Quan he NGUOI_YEU co doi sang giong treu dua khong?\n"

outpath = r"C:\Users\LENOVO\.gemini\antigravity-ide\brain\cba278ee-9be1-4c97-8802-df7cc116e309\llm_persona_test_results.md"
with open(outpath, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nDa xuat file ket qua: {outpath}")
