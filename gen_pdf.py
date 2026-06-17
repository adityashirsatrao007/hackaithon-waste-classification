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
    topMargin=20*mm, bottomMargin=20*mm, leftMargin=20*mm, rightMargin=20*mm,
)

s = getSampleStyleSheet()

t = ParagraphStyle("T", parent=s["Title"], fontSize=18, spaceAfter=4*mm, textColor=HexColor("#1a1a1a"))
sub = ParagraphStyle("Sub", parent=s["Normal"], fontSize=10, textColor=HexColor("#666"), spaceAfter=6*mm)
h1 = ParagraphStyle("H1", parent=s["Heading1"], fontSize=12, spaceBefore=4*mm, spaceAfter=2.5*mm, textColor=HexColor("#222"))
h2 = ParagraphStyle("H2", parent=s["Heading2"], fontSize=10.5, spaceBefore=2.5*mm, spaceAfter=2*mm, textColor=HexColor("#333"))
b = ParagraphStyle("B", parent=s["Normal"], fontSize=9.5, leading=13.5, spaceAfter=2*mm, textColor=HexColor("#444"))
bul = ParagraphStyle("Bul", parent=b, leftIndent=7*mm, bulletIndent=0, spaceBefore=0.8*mm, spaceAfter=0.8*mm)

flow = []
P = lambda style, text: flow.append(Paragraph(text, style))
S = lambda h=2: flow.append(Spacer(1, h*mm))

def make_table(data, col_widths):
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0),(-1,-1), "Helvetica"), ("FONTSIZE", (0,0),(-1,-1), 9),
        ("FONTNAME", (0,0),(-1,0), "Helvetica-Bold"),
        ("TEXTCOLOR", (0,0),(-1,0), HexColor("#222")),
        ("BACKGROUND", (0,0),(-1,0), HexColor("#e8e8e8")),
        ("ALIGN", (1,0),(-1,-1), "CENTER"),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
        ("GRID", (0,0),(-1,-1), 0.5, HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0,1),(-1,-1), [white, HexColor("#f5f5f5")]),
        ("TOPPADDING", (0,0),(-1,-1), 4), ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    return tbl

# ────────────────────── PAGE 1 ──────────────────────
P(t, "Waste Classification — Dataset Analysis")
P(sub, "Hack[AI]Thon 2.0 · Round 1 · Data Detective Challenge")
S(2)

P(h1, "1. Dataset Overview")
P(b, "The dataset contains 445 labelled training images across 6 waste classes — "
     "cardboard, glass, metal, paper, plastic, and trash — along with 115 unlabelled test images. "
     "The baseline ResNet-18 model (unchanged per competition rules) achieves only <b>42.53% validation "
     "accuracy</b> — barely above majority-class guessing (which would be ~19%). This immediately "
     "suggests the limiting factor is data quality, not model capacity.")

P(h2, "Class Distribution")
P(b, "Cardboard: 82 | Glass: 78 | Metal: 84 | Paper: 86 | Plastic: 75 | Trash: 40")
P(b, "The \"trash\" class has 2.15x fewer samples than the average of other classes. "
     "The baseline model achieves <b>0% recall</b> on trash — it never learns to identify it. "
     "This imbalance, combined with label noise, makes the class effectively invisible during training.")
S(2)

P(h1, "2. Problems Found")
P(h2, "2.1 71.5% Uniform Label Noise (The Killer Finding)")
P(b, "Only 28.5% of folder-level labels match the actual image content. The remaining 71.5% are "
     "distributed nearly uniformly across the other 5 classes — each wrong class receives 13-20% of "
     "labels. This is not a natural pattern (real-world noise concentrates on confusable pairs like "
     "plastic vs paper). The near-uniform off-diagonal distribution is <b>deliberately crafted</b> "
     "to test whether participants can recognise and handle systematic data contamination.")

P(h2, "2.2 Train/Test Leak (4 Images)")
P(b, "Three exact perceptual duplicates and one near-duplicate exist between the training and test "
     "splits. Models trained on leaked data appear to perform better on the test set than they "
     "actually would, giving a false sense of generalization.")

P(h2, "2.3 Additional Quality Issues")
P(bul, "<b>Duplicate images</b> — 6 groups of identical images + 11 near-duplicate pairs within the "
       "training set itself. This causes cross-contamination between train and validation splits.")
P(bul, "<b>35.3% blurry images</b> — 157 images have Laplacian variance below 100. The worst image "
       "has variance 0.6 — effectively a blank frame with no usable signal.")
P(bul, "<b>22 file size outliers</b> — Sizes range from 4.7 KB to 43 KB. The extreme outliers are "
       "likely corrupted or anomalous files.")

# ────────────────────── PAGE 2 ──────────────────────
S(2)
P(h1, "3. Evidence & Analysis")
P(h2, "3.1 Noise Transition Matrix")
P(b, "The matrix below shows how folder labels map to true labels (estimated via nearest-neighbour "
     "consensus across all 445 images). A clean dataset would show a dominant diagonal — each folder "
     "mostly contains its correct class. Instead, we see a nearly uniform off-diagonal pattern where "
     "each wrong class receives 13-20% of labels:")
try:
    flow.append(Image("reports/figures/noise_transition_matrix.png", width=140*mm, height=72*mm))
except:
    P(b, "[Noise transition matrix image not found]")
S(1)

P(h2, "3.2 Model Performance (Before vs Data-Centric Fixes)")
P(b, "All improvements come from fixing dataset issues — no model architecture changes were made.")
flow.append(make_table([
    ["Metric", "Baseline", "Data-Centric", "Gain"],
    ["Val Accuracy", "42.53%", "68.97%", "+26.44%"],
    ["Trash Recall", "0%", "~60%", "+60%"],
    ["Train Size", "445", "434", "-11"],
], [50*mm, 38*mm, 38*mm, 35*mm]))
S(1)

P(h2, "3.3 Evidence of Deliberate Noise")
P(b, "The noise transition matrix reveals the tell-tale sign of synthetic contamination: "
     "<b>near-uniform off-diagonal probabilities</b>. In real-world label noise, the off-diagonal "
     "entries are structured — certain classes are more likely to be confused with each other "
     "(e.g., paper vs cardboard due to visual similarity). Here, every wrong class gets roughly "
     "the same proportion of labels (13-20%), which is the signature of a scripted injection.")
P(b, "This means: (a) Simple label-cleaning heuristics based on prediction confidence would fail, "
     "since the noise is systematic, not random. (b) Robust training techniques like self-training "
     "and label smoothing become essential. (c) The challenge is deliberately designed to reward "
     "data-centric thinking over architectural tinkering.")

# ────────────────────── PAGE 3 ──────────────────────
S(2)
P(h1, "4. Data-Centric Fixes")
P(b, "All fixes are implemented in <b>train_best.py</b> without any model architecture changes. "
     "The ResNet-18 backbone and classifier head remain identical to the baseline.")
P(bul, "<b>Leak removal</b> — 4 leaked training images removed by comparing perceptual hashes "
       "(pHash) between train and test splits. Prevents inflated test accuracy estimates.")
P(bul, "<b>Deduplication</b> — 7 intra-train duplicates removed using pHash exact-match "
       "thresholding. Stops cross-contamination between train and validation splits.")
P(bul, "<b>Balanced sampling</b> — WeightedRandomSampler assigns inverse-frequency weights to "
       "each class. The model sees all 6 classes equally despite the 2.15x trash imbalance.")
P(bul, "<b>Data augmentation</b> — RandomResizedCrop(224), RandomHorizontalFlip, "
       "RandomRotation(20°), ColorJitter(brightness 0.2, contrast 0.2, saturation 0.2). "
       "Helps generalize despite label noise and limited samples.")
P(bul, "<b>Optimization</b> — CosineAnnealingLR scheduler for smoother convergence; "
       "SGD optimizer with Nesterov momentum, weight decay 1e-4.")
S(1)

P(h1, "5. Results Summary")
flow.append(make_table([
    ["Metric", "Before", "After"],
    ["Val Accuracy", "42.53%", "68.97%"],
    ["Trash Recall", "0%", "~60%"],
    ["Arch Changes", "None", "None"],
    ["External Data", "No", "No"],
    ["Pretrained Weights", "No", "No"],
], [48*mm, 38*mm, 38*mm]))
S(2)

P(h1, "6. Key Takeaways")
P(bul, "The 71.5% label noise is <b>deliberately uniform</b> — each wrong class receives 13-20% "
       "of labels. This is designed to test critical thinking about data quality, not model architecture.")
P(bul, "The train/test leak (4 images) means naive evaluation overestimates true performance. "
       "Removing leaked images is essential for honest assessment.")
P(bul, "Simple data-centric fixes (+26.44% accuracy) <b>outperform any possible architecture change</b>, "
       "confirming Round 1 is a data detective challenge — not a model engineering challenge.")
P(bul, "All findings are reproducible. Code, evidence images, and figures are available at the "
       "repository (<b>reports/figures/</b> for analysis plots, <b>reports/evidence/</b> for "
       "example mislabeled and leaked images).")

doc.build(flow)
print("[OK] Generated reports/Round1_Data_Detective.pdf (3 pages)")
