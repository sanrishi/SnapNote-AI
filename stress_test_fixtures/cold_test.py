"""Cold test: run 3 untuned fixtures through the gate. No threshold tuning."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["EASYOCR_VERBOSE"] = "0"

from app.services.preprocessor import preprocess
from app.services.ocr_service import (
    read_raw, low_quality_result, _is_english_like
)

FIXTURES = [
    ("7. Biology Prose", "7_biology_prose.png"),
    ("8. Chemistry Equilibrium", "8_chemistry_equilibrium.png"),
    ("9. Dark BST (Unacademy style)", "9_dark_bst.png"),
]

for name, filename in FIXTURES:
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "rb") as f:
        img_bytes = f.read()

    processed = preprocess(img_bytes)
    raw = read_raw(processed)
    decision = low_quality_result(raw)

    tokens = [t.strip() for _, t, _ in raw]
    confs = [conf for _, _, conf in raw]
    avg_conf = sum(confs) / len(confs) if confs else 0
    total_chars = sum(len(t) for t in tokens)

    # Count signals
    sym = sum(1 for t in tokens if any(c in t for c in '}{()[]=+/\\|<>*^'))
    mixed = sum(1 for t in tokens if any(c.isdigit() for c in t) and any(c.isalpha() for c in t))
    end_sym = sum(1 for t in tokens if t and t[-1] in "})]>")
    single_char = sum(1 for t in tokens if len(t) == 1 and t.isalnum())

    alpha_tokens = [t.lower().strip(".,;:!?)}]>\"'-") for t in tokens]
    alpha_tokens = [t for t in alpha_tokens if len(t) >= 3 and t.isalpha()]
    eng_like = sum(1 for t in alpha_tokens if _is_english_like(t)) if alpha_tokens else 0
    avg_len = sum(len(t) for t in alpha_tokens) / len(alpha_tokens) if alpha_tokens else 0

    print(f"=== {name} ===")
    print(f"  low_quality={decision}")
    print(f"  detections={len(tokens)}, chars={total_chars}, avg_conf={avg_conf:.3f}")
    print(f"  symbol_ratio={sym}/{len(tokens)}={sym/len(tokens):.3f}")
    print(f"  mixed_ratio={mixed}/{len(tokens)}={mixed/len(tokens):.3f}")
    print(f"  end_sym_ratio={end_sym}/{len(tokens)}={end_sym/len(tokens):.3f}")
    print(f"  single_char_ratio={single_char}/{len(tokens)}={single_char/len(tokens):.3f}")
    print(f"  eng_like_ratio={eng_like}/{len(alpha_tokens)}={eng_like/len(alpha_tokens):.3f}" if alpha_tokens else "  eng_like_ratio=N/A")
    print(f"  avg_token_len={avg_len:.2f}" if alpha_tokens else "  avg_token_len=N/A")
    print(f"  Expected: {'ESCALATE (formulas)' if name != '7. Biology Prose' else 'STAY OCR (plain prose)'}")
    print(f"  Result: {'PASS' if ((name != '7. Biology Prose') == decision) else 'FAIL'}")
    print()

    # Show first few raw detections
    print("  Raw OCR (first 12):")
    for i, (bbox, text, conf) in enumerate(raw[:12]):
        eng = _is_english_like(text.lower().strip(".,;:!?)}]>\"'-")) if len(text) >= 3 and any(c.isalpha() for c in text) else "?"
        print(f"    [{i:2d}] conf={conf:.3f} len={len(text):2d} eng={eng} {repr(text)}")
    print()
