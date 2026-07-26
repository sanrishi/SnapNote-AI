"""Debug: run OCR on fixture #5, dump raw results + gate decisions."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

os.environ["EASYOCR_VERBOSE"] = "0"

from app.services.preprocessor import preprocess
from app.services.ocr_service import (
    _get_reader, read_raw, raw_to_lines, is_table_layout,
    low_quality_result, _is_english_like
)

IMG = os.path.join(os.path.dirname(__file__), "5_youtube_coaching.png")

with open(IMG, "rb") as f:
    image_bytes = f.read()

processed = preprocess(image_bytes)
raw = read_raw(processed)

print(f"=== OCR returned {len(raw)} detections ===")
print()

tokens = [text.strip() for _, text, _ in raw]
total_chars = sum(len(t) for t in tokens)
confs = [conf for _, _, conf in raw]

print(f"Total characters: {total_chars}")
print(f"Average confidence: {sum(confs)/len(confs):.3f}")
print(f"Low conf ratio (<0.3): {sum(1 for c in confs if c < 0.3)/len(confs):.3f}")
print()

# All detections
print("--- Individual detections ---")
for i, (bbox, text, conf) in enumerate(raw):
    t = text.strip()
    eng = _is_english_like(t.lower().strip(".,;:!?)}]>\"'-")) if len(t) >= 3 and any(c.isalpha() for c in t) else "?"
    print(f"  [{i:2d}] conf={conf:.3f}  len={len(t):2d}  eng={eng}  text={repr(t)}")

print()

# Gate checks
print("--- Gate checks ---")
all_text = " ".join(tokens)
alpha_chars = sum(c.isalpha() or c.isdigit() or c.isspace() for c in all_text)
print(f"1. total_chars < 10: {total_chars} < 10 = {total_chars < 10}")

avg_conf = sum(confs) / len(confs)
print(f"2. avg_conf < 0.4: {avg_conf:.3f} < 0.4 = {avg_conf < 0.4}")

low_conf_ratio = sum(1 for c in confs if c < 0.3) / len(confs)
print(f"3. low_conf_ratio > 0.3: {low_conf_ratio:.3f} > 0.3 = {low_conf_ratio > 0.3}")

alpha_ratio = alpha_chars / len(all_text) if all_text else 0
print(f"4. alpha_chars_ratio < 0.5: {alpha_ratio:.3f} < 0.5 = {alpha_ratio < 0.5}")

single_char = sum(1 for t in tokens if len(t) == 1 and t.isalnum())
sc_ratio = single_char / len(tokens) if tokens else 0
print(f"5. single_char_ratio > 0.3: {sc_ratio:.3f} > 0.3 = {sc_ratio > 0.3}")

# English-likeness
alpha_tokens_clean = [t.lower().strip(".,;:!?)}]>\"'-") for t in tokens]
alpha_tokens_clean = [t for t in alpha_tokens_clean if len(t) >= 3 and t.isalpha()]
if alpha_tokens_clean:
    eng_like = sum(1 for t in alpha_tokens_clean if _is_english_like(t))
    eng_ratio = eng_like / len(alpha_tokens_clean)
    avg_len = sum(len(t) for t in alpha_tokens_clean) / len(alpha_tokens_clean)
else:
    eng_ratio = 0
    avg_len = 0
print(f"6a. english_like_ratio < 0.4: {eng_ratio:.3f} < 0.4 = {eng_ratio < 0.4}")
print(f"6b. avg_token_len < 4.5: {avg_len:.2f} < 4.5 = {avg_len < 4.5}")

# End symbols
end_sym = sum(1 for t in tokens if t and t[-1] in "})]>")
es_ratio = end_sym / len(tokens) if tokens else 0
print(f"7. end_sym_ratio > 0.15: {es_ratio:.3f} > 0.15 = {es_ratio > 0.15}")

# Symbol tokens
symbol_tokens = sum(1 for t in tokens if any(c in t for c in "}{()[]=+/\\|<>*^"))
st_ratio = symbol_tokens / len(tokens) if tokens else 0
print(f"8. symbol_tokens_ratio > 0.5: {st_ratio:.3f} > 0.5 = {st_ratio > 0.5}")

print()
final = low_quality_result(raw)
print(f"=== FINAL DECISION: {'ESCALATE (low quality)' if final else 'OK (keep OCR)'} ===")

# Lines output
lines = raw_to_lines(raw)
print()
print("--- OCR text lines ---")
for l in lines:
    print(f"  {l}")

print()
print(f"Table layout: {is_table_layout(raw)}")
