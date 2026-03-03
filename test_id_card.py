#!/usr/bin/env python3
"""
Test script: OCR Output Reasoning on a Moroccan National ID Card (CNIE)

Simulates a realistic scenario where an OCR system has partially extracted
fields from a Moroccan CNIE but failed on some. The reasoning engine
is asked to resolve the missing/low-confidence fields.
"""

import json
import httpx
import time
import sys

VLLM_URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL = "/opt/fai-llm/models/qwen2.5-7b-awq"

# ─── Simulated OCR output from a Moroccan National ID Card (CNIE) ───
# The OCR system extracted these tokens from the card image.
# Some fields were extracted with high confidence (locked),
# others are missing or ambiguous.

ocr_tokens = [
    {"text": "ROYAUME DU MAROC",     "bbox": {"x": 120, "y": 30,  "w": 200, "h": 18}, "confidence": 0.97, "page": 1, "line": 1, "block": 1},
    {"text": "CARTE NATIONALE",      "bbox": {"x": 130, "y": 52,  "w": 180, "h": 16}, "confidence": 0.96, "page": 1, "line": 2, "block": 1},
    {"text": "D'IDENTITE",           "bbox": {"x": 155, "y": 70,  "w": 120, "h": 16}, "confidence": 0.95, "page": 1, "line": 3, "block": 1},
    {"text": "EL AMRANI",            "bbox": {"x": 220, "y": 120, "w": 140, "h": 18}, "confidence": 0.94, "page": 1, "line": 5, "block": 2},
    {"text": "Nom:",                 "bbox": {"x": 50,  "y": 120, "w": 45,  "h": 16}, "confidence": 0.93, "page": 1, "line": 5, "block": 2},
    {"text": "Prénom:",              "bbox": {"x": 50,  "y": 145, "w": 65,  "h": 16}, "confidence": 0.91, "page": 1, "line": 6, "block": 2},
    {"text": "YOUSSEF",              "bbox": {"x": 220, "y": 145, "w": 110, "h": 18}, "confidence": 0.93, "page": 1, "line": 6, "block": 2},
    {"text": "Date de naissance:",   "bbox": {"x": 50,  "y": 170, "w": 160, "h": 16}, "confidence": 0.90, "page": 1, "line": 7, "block": 2},
    {"text": "15",                   "bbox": {"x": 220, "y": 170, "w": 20,  "h": 18}, "confidence": 0.72, "page": 1, "line": 7, "block": 2},
    {"text": "O3",                   "bbox": {"x": 245, "y": 170, "w": 20,  "h": 18}, "confidence": 0.41, "page": 1, "line": 7, "block": 2},
    {"text": "1992",                 "bbox": {"x": 270, "y": 170, "w": 45,  "h": 18}, "confidence": 0.88, "page": 1, "line": 7, "block": 2},
    {"text": "Lieu de naissance:",   "bbox": {"x": 50,  "y": 195, "w": 155, "h": 16}, "confidence": 0.89, "page": 1, "line": 8, "block": 2},
    {"text": "CASABLANCA",           "bbox": {"x": 220, "y": 195, "w": 140, "h": 18}, "confidence": 0.96, "page": 1, "line": 8, "block": 2},
    {"text": "Sexe:",                "bbox": {"x": 50,  "y": 220, "w": 50,  "h": 16}, "confidence": 0.92, "page": 1, "line": 9, "block": 2},
    {"text": "M",                    "bbox": {"x": 220, "y": 220, "w": 15,  "h": 18}, "confidence": 0.89, "page": 1, "line": 9, "block": 2},
    {"text": "N° CIN:",              "bbox": {"x": 50,  "y": 245, "w": 65,  "h": 16}, "confidence": 0.90, "page": 1, "line": 10, "block": 2},
    {"text": "BK",                   "bbox": {"x": 220, "y": 245, "w": 25,  "h": 18}, "confidence": 0.85, "page": 1, "line": 10, "block": 2},
    {"text": "637421",               "bbox": {"x": 250, "y": 245, "w": 70,  "h": 18}, "confidence": 0.78, "page": 1, "line": 10, "block": 2},
    {"text": "Valable jusqu'au:",    "bbox": {"x": 50,  "y": 275, "w": 150, "h": 16}, "confidence": 0.88, "page": 1, "line": 11, "block": 3},
    {"text": "22.09.2032",           "bbox": {"x": 220, "y": 275, "w": 100, "h": 18}, "confidence": 0.91, "page": 1, "line": 11, "block": 3},
]

# Fields the OCR system already extracted with HIGH confidence (locked — do not override)
locked_fields = {
    "last_name": "EL AMRANI",
    "first_name": "YOUSSEF",
    "place_of_birth": "CASABLANCA",
}

# Fields the OCR system FAILED to extract reliably
missing_fields = ["date_of_birth", "id_number", "gender", "expiry_date"]

# RAG context: past corrections from an analyst for this tenant/document type
rag_context = {
    "field_exemplars": [
        {
            "field_key": "date_of_birth",
            "original_value": "15 O3 1992",
            "corrected_value": "1992-03-15"
        },
        {
            "field_key": "id_number",
            "original_value": "BK 637421",
            "corrected_value": "BK637421"
        },
        {
            "field_key": "date_of_birth",
            "original_value": "O8/12/1987",
            "corrected_value": "1987-12-08"
        }
    ],
    "normalization_rules": [
        {"field_key": "date_of_birth", "output_format": "ISO-8601 (YYYY-MM-DD)"},
        {"field_key": "expiry_date",   "output_format": "ISO-8601 (YYYY-MM-DD)"},
        {"field_key": "id_number",     "output_format": "No spaces, uppercase"},
        {"field_key": "gender",        "output_format": "M or F"}
    ]
}

# ─── Build the prompt (same strategy the FastAPI app will use) ───

system_prompt = """You are an OCR post-processing reasoning engine for identity documents.
Your task is to extract specific fields from OCR tokens that were produced by an upstream OCR system.

Rules:
- Only extract fields listed in missing_fields
- Use locked_fields as ground truth — never override them
- Apply normalization rules from the context exactly
- common OCR misreads: 'O' (letter O) is often '0' (zero), 'l' is often '1', 'S' is often '5'
- If a field cannot be found with confidence > 0.7, set its value to null
- Never invent values that don't appear in the tokens
- Output ONLY valid JSON matching the schema below

Output JSON schema:
{
  "document_type": "string (e.g. NATIONAL_ID_MA)",
  "extracted_fields": {
    "<field_key>": {
      "value": "string or null",
      "confidence": float between 0 and 1,
      "reasoning": "brief explanation of how the value was derived"
    }
  }
}"""

formatted_tokens = "\n".join(
    f"  [{t['line']}:{t['block']}] \"{t['text']}\" (conf={t['confidence']:.2f}, bbox=[{t['bbox']['x']},{t['bbox']['y']},{t['bbox']['w']},{t['bbox']['h']}])"
    for t in ocr_tokens
)

user_prompt = f"""Document type guess: NATIONAL_ID_MA (confidence: 0.92)
Required fields: last_name, first_name, date_of_birth, place_of_birth, gender, id_number, expiry_date
Missing fields (need extraction): {json.dumps(missing_fields)}
Locked fields (already confirmed): {json.dumps(locked_fields, ensure_ascii=False)}

Past analyst corrections for similar documents:
{json.dumps(rag_context['field_exemplars'], indent=2, ensure_ascii=False)}

Normalization rules:
{json.dumps(rag_context['normalization_rules'], indent=2)}

OCR Tokens:
{formatted_tokens}

Extract the missing fields: {json.dumps(missing_fields)}"""


def main():
    print("=" * 70)
    print("  OCR OUTPUT REASONING ENGINE — Test: Moroccan National ID (CNIE)")
    print("=" * 70)
    print()
    print("📄 Scenario:")
    print("   An OCR system scanned a Moroccan CNIE and extracted some fields,")
    print("   but FAILED on: date_of_birth, id_number, gender, expiry_date")
    print("   (the month '03' was misread as 'O3', CIN number had spaces, etc.)")
    print()
    print("🔒 Locked fields (high-confidence OCR, not overridable):")
    for k, v in locked_fields.items():
        print(f"   • {k}: {v}")
    print()
    print("❓ Missing fields to extract via LLM reasoning:")
    for f in missing_fields:
        print(f"   • {f}")
    print()

    # Wait for vLLM to be ready
    print("⏳ Waiting for vLLM server...", end="", flush=True)
    for attempt in range(60):
        try:
            r = httpx.get("http://127.0.0.1:8100/v1/models", timeout=2)
            if r.status_code == 200:
                print(" ✅ Ready!")
                break
        except httpx.ConnectError:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    else:
        print("\n❌ vLLM did not start within 5 minutes. Aborting.")
        sys.exit(1)

    print()
    print("🚀 Sending extraction request to LLM...")
    print("-" * 70)
    start = time.time()

    response = httpx.post(
        VLLM_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        },
        timeout=60,
    )

    elapsed_ms = (time.time() - start) * 1000

    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.text}")
        sys.exit(1)

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    print()
    print("📊 LLM Reasoning Output:")
    print("-" * 70)

    # Try to pretty-print if it's valid JSON
    try:
        parsed = json.loads(content)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(content)

    print("-" * 70)
    print(f"⏱️  Latency: {elapsed_ms:.0f}ms")
    print(f"📈 Tokens: {result['usage']['prompt_tokens']} prompt + {result['usage']['completion_tokens']} completion")
    print()

    # Validate the response
    if "extracted_fields" in content:
        print("✅ Response contains 'extracted_fields' — schema looks correct")
    else:
        print("⚠️  Response may not follow the expected schema")

    # Check for OCR correction reasoning
    if "O3" in content or "03" in content:
        print("✅ LLM recognized the OCR misread (O3 → 03)")

    print()
    print("=" * 70)
    print("  Test complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
