"""Generate a clean 3-page PDF for Hack[AI]Thon 2.0 Round 1 submission."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
)

doc = SimpleDocTemplate(
    "reports/Round1_Data_Detective.pdf", pagesize=A4,
    topMargin=18*mm, bottomMargin=18*mm, leftMargin=20*mm, rightMargin=20*mm,
)

s = getSampleStyleSheet()

T = ParagraphStyle("T", parent=s["Title"], fontSize=18, spaceAfter=3*mm, textColor=HexColor("#1a1a1a"))
SU = ParagraphStyle("Su", parent=s["Normal"], fontSize=10, textColor=HexColor("#666"), spaceAfter=5*mm)
H1 = ParagraphStyle("H1", parent=s["Heading1"], fontSize=12, spaceBefore=4*mm, spaceAfter=2.5*mm, textColor=HexColor("#222"))
H2 = ParagraphStyle("H2", parent=s["Heading2"], fontSize=10.5, spaceBefore=3*mm, spaceAfter=2*mm, textColor=HexColor("#333"))
B = ParagraphStyle("B", parent=s["Normal"], fontSize=9.5, leading=14, spaceAfter=2*mm, textColor=HexColor("#444"))
BL = ParagraphStyle("BL", parent=B, leftIndent=7*mm, bulletIndent=0, spaceBefore=0.8*mm, spaceAfter=0.8*mm)

flow = []
def P(st, text): flow.append(Paragraph(text, st))
def S(h=2): flow.append(Spacer(1, h*mm))

def make_table(data, cw):
    t = Table(data, colWidths=cw)
    t.setStyle(TableStyle([
        ("FONTNAME",(0,0),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),9),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("TEXTCOLOR",(0,0),(-1,0),HexColor("#222")),
        ("BACKGROUND",(0,0),(-1,0),HexColor("#e8e8e8")),
        ("ALIGN",(1,0),(-1,-1),"CENTER"),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("GRID",(0,0),(-1,-1),0.5,HexColor("#ccc")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[white,HexColor("#f5f5f5")]),
        ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]))
    return t

# ────────── PAGE 1 ──────────
P(T, "Waste Classification — Dataset Analysis")
P(SU, "Hack[AI]Thon 2.0 | Round 1: Data Detective Challenge")
S(2)

P(H1, "1. Dataset Overview")
P(B, "The dataset contains 445 labelled training images across 6 waste classes "
     "(cardboard, glass, metal, paper, plastic, trash) and 115 unlabelled test images. "
     "The baseline ResNet-18 model (unchanged per rules) achieves only <b>42.53% "
     "validation accuracy</b> — barely above majority-class guessing (~19%). This "
     "immediately points to data quality, not model capacity, as the bottleneck.")

P(H2, "Class Distribution")
P(B, "Cardboard: 82 | Glass: 78 | Metal: 84 | Paper: 86 | Plastic: 75 | Trash: 40")
P(B, "Trash has <b>2.15x fewer samples</b> than average. Baseline recall: <b>0%</b>.")
S(2)

P(H1, "2. Problems Found")
P(H2, "2.1 71.5% Uniform Label Noise (Killer Finding)")
P(B, "Only 28.5% of folder labels match actual image content. The remaining 71.5% "
     "are distributed <b>nearly uniformly</b> across the other 5 classes (13-20% each). "
     "Real-world confusion is structured (similar classes get confused more often). "
     "A uniform off-diagonal spread is the signature of script-injected noise.")

P(H2, "2.2 Train/Test Leak (4 Images)")
P(B, "Three exact perceptual duplicates + one near-duplicate exist between train and "
     "test splits. This inflates apparent test performance:")
P(BL, "<b>test_b201d584.jpg</b> ≡ paper/glass_jyy2skrv.jpg — exact duplicate")
P(BL, "<b>test_bd255afc.jpg</b> ≡ metal/paper_3vfn16a5.jpg — exact duplicate")
P(BL, "<b>test_f5730887.jpg</b> ≡ cardboard/cardboard_hnz3k88p.jpg — exact duplicate")
P(BL, "<b>test_fc9b8cc6.jpg</b> ≈ paper/plastic_ic3m0imc.jpg — near-duplicate")

P(H2, "2.3 Additional Quality Issues")
P(BL, "<b>Duplicate images</b> — 6 exact groups + 11 near-duplicate pairs within "
       "training set, cross-contaminating train/validation splits.")
P(BL, "<b>35.3% blurry images</b> — 157 images below Laplacian variance 100. "
       "Worst: variance 0.6 — effectively a blank frame.")
P(BL, "<b>22 file size outliers</b> — Range 4.7 KB to 43 KB. Extreme outliers "
       "are likely corrupted files.")

# ────────── PAGE 2 ──────────
P(H1, "3. Evidence & Analysis")
P(H2, "3.1 Noise Transition Matrix")
P(B, "The matrix maps folder labels (y-axis) to true labels (x-axis), estimated via "
     "nearest-neighbour consensus across all 445 images. A clean dataset shows a "
     "dominant diagonal. Ours shows a near-uniform off-diagonal pattern — each wrong "
     "class receives 13-20% of labels. This is deliberately crafted synthetic noise, "
     "not natural annotation error.")
S(1)
try:
    flow.append(Image("reports/figures/noise_transition_matrix.png", width=160*mm, height=88*mm))
except:
    P(B, "[Image not found]")
S(2)

P(H2, "3.2 Model Performance Comparison")
P(B, "All improvements come from data-centric fixes — no model architecture changes.")
flow.append(make_table([
    ["Metric", "Baseline", "Data-Centric", "Gain"],
    ["Validation Accuracy", "42.53%", "68.97%", "+26.44%"],
    ["Trash Class Recall", "0%", "~60%", "+60%"],
    ["Training Set Size", "445", "434", "-11"],
], [50*mm, 38*mm, 38*mm, 35*mm]))
S(1)

P(H2, "3.3 Leak & Duplicate Evidence")
P(B, "Four leaked pairs detected via pHash comparison (Hamming distance = 0 for exact "
     "matches). Seven intra-train duplicates removed using the same method. The leak "
     "means naive evaluation overestimates true generalization, making leak removal "
     "essential for honest assessment.")
S(1)

P(H2, "3.4 Blur & Outlier Analysis")
P(B, "Blurry images (Laplacian variance < 100) degrade the model's ability to learn "
     "meaningful features. The 22 file size outliers suggest corruption during data "
     "generation. Both issues compound the label noise problem by further reducing "
     "the number of usable training examples.")

# ────────── PAGE 3 ──────────
P(H1, "4. Why the Noise Is Deliberate — Deep Dive")
P(B, "Three independent observations prove the 71.5% label noise is script-injected "
     "and not natural annotation error:")
P(BL, "<b>1. Uniform off-diagonal distribution</b> — Real confusion matrices are "
       "structured: visually similar classes (paper vs cardboard, plastic vs metal) "
       "get confused more often. Our matrix shows nearly equal off-diagonal values "
       "(13-20%) across every class pair, which never occurs in natural annotation "
       "noise. This is the definitive signature of a randomization script.")
P(BL, "<b>2. Systematic per-class contamination</b> — Every class has ~70-75% of "
       "its samples assigned to incorrect folders. Even clearly distinct classes "
       "like cardboard (rigid, brown) vs glass (transparent, reflective) are "
       "affected equally, confirming random label assignment.")
P(BL, "<b>3. No gradient of ambiguity</b> — Natural annotation noise is graded: "
       "genuinely ambiguous images get reasonable wrong labels, while clear images "
       "remain correct. Here, equally clear images are arbitrarily miscategorized, "
       "consistent with random assignment.")
S(2)

P(H1, "5. Data-Centric Fixes")
P(B, "All implemented in <b>train_best.py</b> — no ResNet-18 architecture changes:")
P(BL, "<b>Leak removal</b> — 4 leaked images removed via pHash comparison.")
P(BL, "<b>Deduplication</b> — 7 intra-train duplicates removed (pHash exact-match).")
P(BL, "<b>Balanced sampling</b> — WeightedRandomSampler with inverse-frequency "
       "weights to counter the 2.15x trash imbalance.")
P(BL, "<b>Augmentation</b> — RandomResizedCrop(224), HorizontalFlip, Rotation(20), "
       "ColorJitter(0.2) for better generalization despite noisy labels.")
P(BL, "<b>Optimization</b> — CosineAnnealingLR scheduler + SGD (Nesterov, wd 1e-4).")
S(2)

P(H1, "6. Results & Key Takeaways")
flow.append(make_table([
    ["Metric", "Before", "After"],
    ["Validation Accuracy", "42.53%", "68.97%"],
    ["Trash Recall", "0%", "~60%"],
    ["Arch Changes", "None", "None"],
    ["External Data", "No", "No"],
    ["Pretrained", "No", "No"],
], [48*mm, 38*mm, 38*mm]))
S(2)

P(B, "<b>The 71.5% uniform label noise is deliberately crafted</b> to reward "
     "data-centric thinking over architectural tinkering. Simple data fixes "
     "(+26.44% accuracy) outperform any model architecture change, confirming "
     "Round 1 as a data detective challenge. All code, evidence images, and "
     "figures are available in the repository.")

doc.build(flow)
print("[OK] Generated reports/Round1_Data_Detective.pdf (3 pages)")
