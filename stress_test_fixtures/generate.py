import os
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))

def get_font(size):
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\arial.ttf", size)
    except:
        return ImageFont.load_default()

def get_deva(size):
    try:
        return ImageFont.truetype("C:\\Windows\\Fonts\\Nirmala.ttf", size)
    except:
        return get_font(size)

def dt(draw, xy, text, font, fill="black"):
    draw.text(xy, text, fill=fill, font=font)

F = get_font
FD = get_deva

# ── 1. Stats probability table ──
img = Image.new("RGB", (900, 500), "white")
d = ImageDraw.Draw(img)
dt(d, (30, 10), "Probability Distribution Table", F(22))
dt(d, (30, 45), "Distribution  |  Mean (u)  |  Variance (s^2)  |  MGF", F(16))
rows = [
    ("Binomial(n,p)", "np", "np(1-p)", "(1-p+pe^t)^n"),
    ("Poisson(l)", "l", "l", "exp(l(e^t-1))"),
    ("Normal(u,s^2)", "u", "s^2", "exp(ut+s^2t^2/2)"),
    ("Exponential(l)", "1/l", "1/l^2", "l/(l-t)"),
]
y = 80
for r in rows:
    dt(d, (30, y), "  ".join(f"{c:<25}" for c in r), F(14))
    y += 45
for yy in range(75, 80 + 4 * 45 + 5, 45):
    d.line([(20, yy), (880, yy)], fill="gray", width=1)
d.line([(250, 40), (250, y - 45)], fill="gray", width=1)
d.line([(450, 40), (450, y - 45)], fill="gray", width=1)
d.line([(660, 40), (660, y - 45)], fill="gray", width=1)
dt(d, (30, y + 10), "Note: MGF = Moment Generating Function. Used for deriving moments.", F(12), fill="gray")
img.save(os.path.join(OUT, "1_stats_table.png"))
print("1 done")

# ── 2. Engineering flowchart ──
img = Image.new("RGB", (800, 600), "white")
d = ImageDraw.Draw(img)
dt(d, (250, 10), "PID Controller Flow", F(22))


def box(d, x, y, w, h, text, font=F(13)):
    d.rectangle([x, y, x + w, y + h], outline="black", width=2, fill="#e8f0fe")
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        lw = d.textlength(ln, font=font)
        dt(d, (x + (w - lw) // 2, y + (h - 12 * len(lines)) // 2 + i * 14), ln, font)


box(d, 300, 50, 180, 40, "Set Point r(t)")
box(d, 300, 130, 180, 40, "Error e(t)")
box(d, 130, 220, 160, 50, "Proportional\nKp x e(t)")
box(d, 330, 220, 160, 50, "Integral\nKi x Integ e(t)dt")
box(d, 530, 220, 160, 50, "Derivative\nKd x de/dt")
box(d, 300, 330, 180, 40, "Summation S")
box(d, 300, 420, 180, 40, "Plant G(s)")
box(d, 300, 500, 180, 40, "Output y(t)")

d.line([(390, 90), (390, 130)], fill="black", width=2)
d.line([(210, 240), (300, 240)], fill="black", width=2)
d.line([(390, 170), (390, 200)], fill="black", width=2)
d.line([(490, 240), (530, 240)], fill="black", width=2)
d.line([(300, 270), (210, 270), (210, 240)], fill="black", width=2)
d.line([(390, 270), (390, 330)], fill="black", width=2)
d.line([(390, 370), (390, 420)], fill="black", width=2)
d.line([(390, 460), (390, 500)], fill="black", width=2)
d.line([(480, 520), (100, 520), (100, 240), (130, 240)], fill="black", width=2)
dt(d, (40, 515), "Feedback (sensor)", F(11), fill="gray")
for tip in [(390, 130), (300, 240), (390, 200), (530, 240), (210, 240), (390, 330), (390, 420), (390, 500), (130, 240)]:
    d.polygon([tip, (tip[0] - 6, tip[1] - 8), (tip[0] + 6, tip[1] - 8)], fill="black")
img.save(os.path.join(OUT, "2_flowchart.png"))
print("2 done")

# ── 3. Math formulas ──
img = Image.new("RGB", (900, 500), "white")
d = ImageDraw.Draw(img)
dt(d, (20, 10), "Advanced Engineering Mathematics", F(22))
dt(d, (20, 50), "1. Quadratic Formula:", F(16))
dt(d, (40, 78), "x = (-b +/- sqrt(b^2 - 4ac)) / (2a)", F(16))
dt(d, (20, 115), "2. Summation Notation:", F(16))
dt(d, (40, 143), "Sum_{i=1}^{n} i = n(n+1)/2", F(16))
dt(d, (20, 180), "3. Standard Normal Distribution:", F(16))
dt(d, (40, 208), "f(z) = (1/sqrt(2pi)) x e^(-z^2/2)", F(16))
dt(d, (20, 245), "4. Taylor Series Expansion:", F(16))
dt(d, (40, 273), "f(x) = Sum_{n=0}^{inf} f^(n)(a)/n! x (x-a)^n", F(16))
dt(d, (20, 310), "5. Fourier Transform:", F(16))
dt(d, (40, 338), "F(w) = Integ_{-inf}^{inf} f(t) x e^{-jwt} dt", F(16))
dt(d, (20, 375), "6. Greek Symbols used in Statistics:", F(16))
dt(d, (40, 403), "u (mean) | s (std dev) | S (sum) | p (pi) | a (significance) | b (error)", F(16))
dt(d, (40, 435), "l (Poisson rate) | r (correlation) | chi^2 (chi-square) | theta (angle)", F(16))
img.save(os.path.join(OUT, "3_math_formulas.png"))
print("3 done")

# ── 4. Hindi/Hinglish coaching slide (Devanagari + English) ──
img = Image.new("RGB", (900, 550), "white")
d = ImageDraw.Draw(img)
fd = FD(20)
fe = F(16)
# Using Nirmala UI font which supports Devanagari
dt(d, (30, 15), "\u092d\u094c\u0924\u093f\u0915 \u0935\u093f\u091c\u094d\u091e\u093e\u0928 - Physics Class 12", fd, fill="#1a237e")
dt(d, (30, 50), "\u0935\u093f\u0937\u092f: \u0935\u093f\u0926\u094d\u092f\u0941\u0924 \u0927\u093e\u0930\u093e (Electric Current)", fd, fill="#283593")
dt(d, (30, 90), "Current = I = Q/t (\u0915\u0942\u0932\u0949\u092e/\u0938\u0947\u0915\u0902\u0921 = \u090f\u092e\u094d\u092a\u093f\u092f\u0930)", fd)
dt(d, (30, 125), "Ohm's Law: V = IR", fe)
dt(d, (30, 155), "\u091c\u0939\u093e\u0901 R = \u092a\u094d\u0930\u0924\u093f\u0930\u094b\u0927 (Resistance), unit = Ohm (\u092e\u0948\u0924\u094d\u0930\u093f\u0915\u094d\u0938 \u092e\u0947\u0902 \u0928\u0939\u0940\u0902)", fd)
dt(d, (30, 195), "Resistivity Formula:", fe)
dt(d, (50, 225), "\u03c1 = RA/L", fe)
dt(d, (50, 255), "\u091c\u0939\u093e\u0901 A = area of cross-section, L = length", fd)
dt(d, (30, 295), "Series vs Parallel Combination:", fe)
dt(d, (50, 325), "Series: R_eq = R1 + R2 + R3 + ...", fe)
dt(d, (50, 355), "Parallel: 1/R_eq = 1/R1 + 1/R2 + 1/R3 + ...", fe)
dt(d, (30, 400), "Important Numerical:", fe)
dt(d, (50, 430), "\u092a\u094d\u0930\u0936\u094d\u0928: 10\u03a9 \u0915\u093e \u092a\u094d\u0930\u0924\u093f\u0930\u094b\u0927 5V \u0915\u0940 \u092c\u0948\u091f\u0930\u0940 \u0938\u0947 \u091c\u094b\u0921\u093c\u093e \u0917\u092f\u093e \u0939\u0948\u0964 \u0935\u093f\u0926\u094d\u092f\u0941\u0924 \u0927\u093e\u0930\u093e \u091c\u094d\u091e\u093e\u0924 \u0915\u0930\u094b\u0964", fd)
dt(d, (50, 465), "\u0939\u0932: I = V/R = 5/10 = 0.5 A", fe)
dt(d, (30, 505), "\u0928\u094b\u091f: \u092f\u093e\u0926 \u0930\u0916\u0947\u0902 \u2014 Current always flows from +ve to -ve terminal", fd, fill="#c62828")
img.save(os.path.join(OUT, "4_hindi_coaching.png"))
print("4 done")

# ── 5. YouTube coaching-style slide ──
img = Image.new("RGB", (900, 550), "#f5f5ff")
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 900, 55], fill="#1a237e")
dt(d, (30, 12), "JEE Main 2026 | Physics | Electrostatics", F(22), fill="white")
dt(d, (30, 40), "Lecture 12: Electric Potential & Capacitance", F(15), fill="#bbdefb")
dt(d, (30, 75), "Key Concepts:", F(18), fill="#1a237e")
dt(d, (30, 105), ">> Electric Potential (V) = Work done per unit charge = W/q", F(15))
dt(d, (30, 135), ">> Potential due to point charge: V = kQ/r", F(15))
dt(d, (30, 165), ">> Equipotential surfaces - no work done along them", F(15))
dt(d, (30, 205), "Capacitance Formula:", F(18), fill="#1a237e")
d.rectangle([25, 200, 450, 235], outline="#1a237e", width=1)
dt(d, (30, 240), "C = Q/V  (Farads)", F(20))
dt(d, (30, 275), "Parallel Plate: C = epsilon_0 x A / d", F(16))
dt(d, (30, 310), "Energy stored: U = 1/2 CV^2 = 1/2 QV = Q^2/2C", F(16))
dt(d, (30, 350), "Important Combinations:", F(18), fill="#1a237e")
dt(d, (30, 380), "Series: 1/C_eq = 1/C1 + 1/C2 + 1/C3", F(15))
dt(d, (30, 410), "Parallel: C_eq = C1 + C2 + C3 + ...", F(15))
dt(d, (30, 455), "Home Work:", F(16), fill="#c62828")
dt(d, (30, 485), "Q: Find equivalent capacitance between A and B if C1=2uF, C2=4uF, C3=6uF in series.", F(14))
d.rectangle([0, 530, 900, 550], fill="#1a237e")
dt(d, (250, 534), "Subscribe: Physics Wallah | PW App | 2026", F(12), fill="#bbdefb")
img.save(os.path.join(OUT, "5_youtube_coaching.png"))
print("5 done")

# ── 6. Plain prose slide (no formulas, no tables, no symbols) ──
img = Image.new("RGB", (900, 550), "white")
d = ImageDraw.Draw(img)
dt(d, (30, 15), "Introduction to Economics", F(22), fill="#1a237e")
dt(d, (30, 50), "Chapter 1: Basic Concepts", F(18), fill="#283593")
dt(d, (30, 85), "Economics is the study of how people make choices under", F(16))
dt(d, (30, 110), "conditions of scarcity. Scarcity means that society has", F(16))
dt(d, (30, 135), "limited resources and cannot produce all the goods and", F(16))
dt(d, (30, 160), "services that people wish to have.", F(16))
dt(d, (30, 200), "The fundamental economic problem is the allocation of", F(16))
dt(d, (30, 225), "scarce resources among competing uses. Every economy", F(16))
dt(d, (30, 250), "must answer three basic questions:", F(16))
dt(d, (50, 285), "1. What to produce?", F(16))
dt(d, (50, 315), "2. How to produce?", F(16))
dt(d, (50, 345), "3. For whom to produce?", F(16))
dt(d, (30, 390), "Opportunity cost is the value of the next best alternative", F(16))
dt(d, (30, 415), "that is given up when a choice is made. It is a key concept", F(16))
dt(d, (30, 440), "in economics that helps in understanding trade-offs.", F(16))
dt(d, (30, 480), "Microeconomics studies individual units like households", F(16))
dt(d, (30, 505), "and firms. Macroeconomics studies the economy as a whole.", F(16))
img.save(os.path.join(OUT, "6_plain_prose.png"))
print("6 done")

# ── 7. Biology prose slide (plain text, no formulas) ──
img = Image.new("RGB", (900, 500), "#fafafa")
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 900, 45], fill="#00695c")
dt(d, (20, 10), "NEET Biology | Cell: The Unit of Life", F(18), fill="white")
dt(d, (20, 60), "Definition of Cell:", F(18), fill="#00695c")
dt(d, (20, 90), "The cell is the fundamental structural and functional unit of all", F(16))
dt(d, (20, 115), "living organisms. Cells are often called the building blocks of life.", F(16))
dt(d, (20, 145), "Key Discoveries:", F(18), fill="#00695c")
dt(d, (20, 175), "Robert Hooke (1665) first observed cells in a cork slice using a", F(16))
dt(d, (20, 200), "primitive microscope. Anton van Leeuwenhoek observed living", F(16))
dt(d, (20, 225), "cells in pond water and called them animalcules.", F(16))
dt(d, (20, 255), "Cell Theory:", F(18), fill="#00695c")
dt(d, (20, 285), "1. All living organisms are composed of one or more cells.", F(16))
dt(d, (20, 310), "2. The cell is the basic unit of structure and organization.", F(16))
dt(d, (20, 335), "3. All cells arise from pre-existing cells.", F(16))
dt(d, (20, 365), "Two main types of cells:", F(18), fill="#00695c")
dt(d, (20, 395), "Prokaryotic cells lack a nucleus and membrane-bound organelles.", F(16))
dt(d, (20, 420), "Eukaryotic cells have a nucleus and membrane-bound organelles.", F(16))
dt(d, (20, 450), "Examples: Bacteria are prokaryotes. Plants and animals are eukaryotes.", F(16))
img.save(os.path.join(OUT, "7_biology_prose.png"))
print("7 done")

# ── 8. Chemistry formula slide (equations, periodic trends, symbols) ──
img = Image.new("RGB", (900, 500), "white")
d = ImageDraw.Draw(img)
dt(d, (20, 10), "JEE Chemistry | Chemical Equilibrium", F(20), fill="#b71c1c")
dt(d, (20, 45), "Equilibrium Constant:  Kc = [C]^c [D]^d / [A]^a [B]^b", F(16))
dt(d, (20, 80), "For reaction: aA + bB <=> cC + dD", F(16))
dt(d, (20, 115), "Kp = Kc (RT)^(delta_n)   where delta_n = (c+d) - (a+b)", F(16))
dt(d, (20, 150), "Le Chatelier's Principle:", F(18), fill="#b71c1c")
dt(d, (20, 180), "If a system at equilibrium is disturbed by changing", F(16))
dt(d, (20, 205), "temperature, pressure, or concentration, the system", F(16))
dt(d, (20, 230), "shifts to counteract the change.", F(16))
dt(d, (20, 265), "Reaction Quotient:  Qc = [C]^c [D]^d / [A]^a [B]^b", F(16))
dt(d, (20, 300), "If Qc < Kc, reaction proceeds forward.", F(16))
dt(d, (20, 335), "If Qc > Kc, reaction proceeds backward.", F(16))
dt(d, (20, 370), "Relation between Kp and Kc:", F(18), fill="#b71c1c")
dt(d, (20, 400), "Kp = Kc (RT)^(delta_n) — same formula", F(16))
dt(d, (20, 435), "Units: Kc has units of (mol/L)^(delta_n)", F(16))
dt(d, (20, 465), "Kp has units of (atm)^(delta_n)", F(16))
img.save(os.path.join(OUT, "8_chemistry_equilibrium.png"))
print("8 done")

# ── 9. Dark-theme coaching slide (different visual style, like Unacademy) ──
img = Image.new("RGB", (900, 500), "#1a1a2e")
d = ImageDraw.Draw(img)
d.rectangle([0, 0, 900, 50], fill="#16213e")
dt(d, (20, 12), "UNACADEMY  |  GATE 2026  |  Computer Science", F(16), fill="#e94560")
dt(d, (20, 65), "Data Structures: Binary Search Trees", F(22), fill="#f5f5f5")
dt(d, (20, 105), "Properties of BST:", F(18), fill="#e94560")
dt(d, (30, 135), "Left subtree contains only nodes with keys less than", F(15), fill="#ccc")
dt(d, (30, 160), "the parent node's key.", F(15), fill="#ccc")
dt(d, (30, 185), "Right subtree contains only nodes with keys greater", F(15), fill="#ccc")
dt(d, (30, 210), "than the parent node's key.", F(15), fill="#ccc")
dt(d, (20, 245), "Common BST Operations:", F(18), fill="#e94560")
dt(d, (30, 275), "Search: O(h) where h = height of tree", F(15), fill="#ccc")
dt(d, (30, 300), "Insert: O(h)", F(15), fill="#ccc")
dt(d, (30, 325), "Delete: O(h) — three cases to handle", F(15), fill="#ccc")
dt(d, (20, 360), "Tree Traversals:", F(18), fill="#e94560")
dt(d, (30, 390), "In-order (LNR): outputs sorted order", F(15), fill="#ccc")
dt(d, (30, 415), "Pre-order (NLR): creates copy of tree", F(15), fill="#ccc")
dt(d, (30, 440), "Post-order (LRN): deletes the tree", F(15), fill="#ccc")
dt(d, (20, 470), "Worst-case: O(n) for skewed tree, O(log n) for balanced", F(14), fill="#e94560")
img.save(os.path.join(OUT, "9_dark_bst.png"))
print("9 done")

print("All 9 images created.")
