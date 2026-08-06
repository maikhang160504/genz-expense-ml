import sys
import re

# 1. Update llm_intent_handler.py
with open('src/nlu/llm_intent_handler.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add import re if not present
if 'import re\n' not in content:
    content = content.replace('def _build_relationship_addition(', 'import re\n\ndef _build_relationship_addition(')

# Fix relationship_addition regex
content = content.replace('if any(kw in text_lower for kw in cha_me_keywords):',
                          'if any(re.search(rf"\\\\b{kw}\\\\b", text_lower) for kw in cha_me_keywords):')
content = content.replace('if any(kw in text_lower for kw in nguoi_yeu_keywords):',
                          'if any(re.search(rf"\\\\b{kw}\\\\b", text_lower) for kw in nguoi_yeu_keywords):')
content = content.replace('is_expense = any(kw in text_lower for kw in',
                          'is_expense = any(re.search(rf"\\\\b{kw}\\\\b", text_lower) for kw in')

# Fix build_system_prompt
old_bsp = '''def _build_system_prompt(intent: str, nlg_persona: str | None, text: str, is_rag: bool = False) -> str:'''
new_bsp = '''def _build_system_prompt(intent: str, nlg_persona: str | None, text: str, is_rag: bool = False, caller_context: str | None = None) -> str:'''
content = content.replace(old_bsp, new_bsp)

old_blocks = '''    persona_block = _build_persona_addition(nlg_persona, prompts)
    relationship_block = _build_relationship_addition(text, prompts)'''
new_blocks = '''    if caller_context == "bill":
        persona_block = ""
        relationship_block = ""
    else:
        persona_block = _build_persona_addition(nlg_persona, prompts)
        relationship_block = _build_relationship_addition(text, prompts)'''
content = content.replace(old_blocks, new_blocks)

# Fix run_llm_nlu_v2 signature and call
content = re.sub(r'(def run_llm_nlu_v2\([^)]+override_prompt:\s*str\s*\|\s*None\s*=\s*None,)([^)]*\)\s*->\s*dict\[str,\s*Any\]:)',
                 r'\g<1> caller_context: str | None = None, \g<2>', content)
content = content.replace('system_prompt = _build_system_prompt(intent, nlg_persona, text, is_rag=is_rag)',
                          'system_prompt = _build_system_prompt(intent, nlg_persona, text, is_rag=is_rag, caller_context=caller_context)')

with open('src/nlu/llm_intent_handler.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated llm_intent_handler successfully")
