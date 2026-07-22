import urllib.request, json, datetime

MODAL_URL = 'https://maikhang160504--expense-ocr-nlu-fastapi-app-dev.modal.run'

# Map uppercase verbal_style -> lowercase persona key trong prompts.json
PERSONA_MAP = {
    'DUI_DE': 'dui_de',
    'DAN_DOI': 'dan_doi',
    'KHO_TINH': 'kho_tinh',
    'NGOT_NGAO': 'ngot_ngao',
}

test_cases = [
    {'text': 'an trua bun bo 45k', 'verbal_style': 'DUI_DE'},
    {'text': 'mua iPhone 25 trieu', 'verbal_style': 'DAN_DOI'},
    {'text': 'mua cafe Starbucks 120k', 'verbal_style': 'KHO_TINH'},
    {'text': 'di spa 800k', 'verbal_style': 'NGOT_NGAO'},
    {'text': 'mua qua tang nguoi yeu 300k', 'verbal_style': 'DUI_DE'},
    {'text': 'bieu me 500k', 'verbal_style': 'DAN_DOI'},
    {'text': 'nhan luong thang nay 8 trieu', 'verbal_style': 'NGOT_NGAO'},
    {'text': 'hom nay tieu nhieu qua, co cach nao tiet kiem khong?', 'verbal_style': 'KHO_TINH'},
]

results = []
for tc in test_cases:
    nlg_persona = PERSONA_MAP.get(tc['verbal_style'], tc['verbal_style'].lower())
    payload = json.dumps({
        'text': tc['text'],
        'nlg_persona': nlg_persona,   # TOP-LEVEL field trong NLURequest schema
        'profile': {
            'username': 'Khang',
        },
        'run_llm': True
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{MODAL_URL}/api/v1/nlu/infer",
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    print(f"Testing [{tc['verbal_style']}]: {tc['text']!r} ...", flush=True)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode('utf-8'))
            resp = (
                data.get('nlg_response') or
                data.get('response') or
                (data.get('nlg') or {}).get('response') or
                '(Khong co phan hoi)'
            )
            intent = data.get('intent', '?')
            emotion = data.get('mimo_emotion') or data.get('llm_emotion') or data.get('emotion') or '?'
            nlg_persona_used = data.get('nlg_persona', '?')
            print(f"  -> [{intent}|{emotion}|persona={nlg_persona_used}] {resp}")
            results.append({
                'verbal_style': tc['verbal_style'],
                'nlg_persona': nlg_persona,
                'nlg_persona_used': nlg_persona_used,
                'input': tc['text'],
                'intent': intent,
                'emotion': emotion,
                'response': resp,
                'ok': True
            })
    except Exception as e:
        print(f"  -> ERROR: {e}")
        results.append({
            'verbal_style': tc['verbal_style'],
            'nlg_persona': nlg_persona,
            'input': tc['text'],
            'error': str(e),
            'ok': False
        })

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
outpath = r'C:\Users\LENOVO\.gemini\antigravity-ide\brain\cba278ee-9be1-4c97-8802-df7cc116e309\llm_persona_test_results.md'

ok_count = sum(1 for r in results if r['ok'])
persona_ok = sum(1 for r in results if r.get('ok') and r.get('nlg_persona_used') not in ('?', None))

md = f"# Ket qua Test LLM Persona Prompt (Qwen 2.5 tren Modal)\n\n"
md += f"**Thoi gian:** {now}  \n"
md += f"**Endpoint:** `{MODAL_URL}`  \n"
md += f"**Model:** Qwen2.5-14B-Instruct (A10G GPU) fallback Gemini  \n\n---\n\n"
md += f"## Tong quan: {ok_count}/{len(results)} thanh cong, {persona_ok}/{ok_count} co nlg_persona dung\n\n"

md += "| # | Verbal Style | nlg_persona gui | nlg_persona nhan | Input | Intent | Emotion | Phan hoi AI |\n"
md += "|---|---|---|---|---|---|---|---|\n"
for i, r in enumerate(results, 1):
    if r['ok']:
        persona_recv = r.get('nlg_persona_used', '?')
        persona_match = "OK" if persona_recv == r['nlg_persona'] else "LECH"
        md += f"| {i} | `{r['verbal_style']}` | `{r['nlg_persona']}` | `{persona_recv}` ({persona_match}) | {r['input']} | {r['intent']} | {r['emotion']} | {r['response'][:80]}... |\n"
    else:
        md += f"| {i} | `{r['verbal_style']}` | `{r['nlg_persona']}` | - | {r['input']} | - | - | LOI: {r.get('error','')} |\n"

md += "\n---\n\n## Chi tiet phan hoi AI theo tung Persona\n\n"
for i, r in enumerate(results, 1):
    md += f"### {i}. [{r['verbal_style']}] — `{r['input']}`\n"
    if r['ok']:
        persona_recv = r.get('nlg_persona_used', '?')
        persona_match = "DUNG" if persona_recv == r['nlg_persona'] else f"LECH (nhan duoc: {persona_recv})"
        md += f"- **nlg_persona gui:** `{r['nlg_persona']}`\n"
        md += f"- **nlg_persona nhan:** `{persona_recv}` — **{persona_match}**\n"
        md += f"- **Intent:** `{r['intent']}`\n"
        md += f"- **Emotion:** `{r['emotion']}`\n"
        md += f"- **Phan hoi Mimo:**\n\n> {r['response']}\n\n"
    else:
        md += f"- **Loi:** `{r['error']}`\n\n"

md += "---\n\n## Danh gia Persona\n\n"
md += "- [ ] DUI_DE: vui ve, nang luong cao, dung slang dui de?\n"
md += "- [ ] DAN_DOI: cang nhang, xeo xat, buc boi?\n"
md += "- [ ] KHO_TINH: nghiem khac, ky luat, phan tich lanh lung?\n"
md += "- [ ] NGOT_NGAO: ngot ngao, chua lanh, dong vien?\n"
md += "- [ ] NGUOI_YEU keyword: doi sang giong treu dua?\n"
md += "- [ ] CHA_ME keyword: doi sang giong am ap, tu hao?\n"

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(md)
print(f"\n=== Da ghi ket qua: {outpath} ===")
