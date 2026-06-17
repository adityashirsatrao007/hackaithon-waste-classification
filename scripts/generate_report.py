from fpdf import FPDF
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FIGURES = BASE / "reports" / "figures"

class Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150)
        self.cell(0, 4, "Hack[AI]Thon 2.0 | Data Detective Challenge | Round 1", align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")

    def sec(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30)
        self.cell(0, 7, title)
        self.ln(4)
        self.set_draw_color(200, 50, 50)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def txt(self, t):
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(50)
        self.multi_cell(0, 4, t)
        self.ln(1)

    def blt(self, title, desc):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(50)
        x = self.get_x()
        self.cell(4, 4, "-")
        self.cell(0, 4, title)
        self.ln(4)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(70)
        self.set_x(x + 6)
        self.multi_cell(0, 3.5, desc)
        self.ln(0.5)

    def tbl(self, header, rows, cw):
        for i, row in enumerate([header] + rows):
            self.set_font("Helvetica", "B" if i == 0 else "", 8)
            self.set_fill_color(235, 235, 235) if i == 0 else self.set_fill_color(255, 255, 255)
            for j, v in enumerate(row):
                self.cell(cw[j], 4.5, str(v), border=1, fill=True, align="C")
            self.ln()


pdf = Report()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=14)

# ── PAGE 1 ──
pdf.add_page()

pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(30)
pdf.cell(0, 10, "Data Detective Challenge")
pdf.ln(1)
pdf.set_font("Helvetica", "", 8)
pdf.set_text_color(120)
pdf.cell(0, 5, "Waste Classification Dataset Audit | Hack[AI]Thon 2.0 | Round 1")
pdf.ln(8)

pdf.sec("1  Dataset Overview")
pdf.txt(
    "560 images (445 train, 115 test), 6 classes: cardboard, glass, metal, paper, plastic, trash. "
    "Filname format: {true_label}_{random_id}.jpg. ResNet-18 from scratch, 10 epochs. "
    "Class counts: paper=86, metal=84, cardboard=82, glass=78, plastic=75, trash=40 "
    "(2.15x imbalance)."
)

pdf.sec("2  Problems Found")
pdf.txt("Six issues identified:")

pdf.blt("Label Noise (71.5%)",
    "318/445 images in wrong folder. Filename prefixes also unreliable: model trained on "
    "corrected labels achieved 20.22% (random=16.7%), loss stuck at 1.78.")
pdf.blt("Blurry Images (35%)",
    "157 images with Laplacian variance < 100. Minimum variance: 0.6.")
pdf.blt("Class Imbalance",
    "Trash (40) vs paper (86) = 2.15x ratio. Per-class accuracy shows trash near zero.")
pdf.blt("Duplicates",
    "6 exact-MD5 duplicate groups, 11 near-duplicate pHash pairs.")
pdf.blt("Outliers",
    "22 file-size outliers outside 1.5x IQR. No corrupted images.")

# ── PAGE 2 ──
pdf.add_page()
pdf.sec("3  Evidence")

cm = FIGURES / "confusion_matrix.png"
if cm.exists():
    pdf.image(str(cm), x=12, w=82)
    pdf.set_xy(99, pdf.get_y() - 44)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(70)
    pdf.multi_cell(92, 4,
        "Confusion matrix: predictions concentrate on diagonal for "
        "cardboard/glass but scatter for paper -- the class with most "
        "folder-label errors."
    )

pdf.ln(14)
umap_f = FIGURES / "umap_by_folder.png"
umap_t = FIGURES / "umap_by_truelabel.png"
if umap_f.exists() and umap_t.exists():
    pdf.image(str(umap_f), x=8, w=95)
    pdf.image(str(umap_t), x=105, w=95)
    pdf.ln(44)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(80)
    pdf.cell(0, 4, "Left: UMAP by folder label.    Right: UMAP by filename prefix.")
    pdf.ln(6)

conf = FIGURES / "confidence_hist.png"
if conf.exists():
    pdf.image(str(conf), x=18, w=72)
    pdf.set_xy(96, pdf.get_y() - 34)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(70)
    pdf.multi_cell(92, 4,
        "Confidence distribution: most predictions in 40-60% range. "
        "Model lacks high-confidence discrimination, consistent with "
        "noisy training labels."
    )



# ── PAGE 3 ──
pdf.add_page()
pdf.sec("4  Suggested Fixes")

pdf.blt("Re-label dataset [HIGH]",
    "Both folder labels and filename prefixes unreliable. Use confident "
    "learning (CleanLab) or manual verification.")
pdf.blt("Stratified splits [HIGH]",
    "Stratified sampling to maintain class proportions in splits.")
pdf.blt("Down-weight blurry [MED]",
    "Lower sample weights for images with Laplacian variance < 100.")
pdf.blt("Augment trash class [MED]",
    "Aggressive augmentation or class-weighted loss for 2.15x imbalance.")
pdf.blt("Remove duplicates [LOW]",
    "Deduplicate exact MD5 matches and near-duplicate pairs.")
pdf.blt("Noise-robust training [MED]",
    "Label smoothing or peer loss for remaining label noise.")

pdf.sec("5  Expected Impact")
pdf.txt("Ablation study: same ResNet-18, 10 epochs, two label versions.")

pdf.tbl(
    ("Setup", "Best Val Acc", "Loss"),
    [
        ("Folder labels (buggy)", "44.94%", "Decreasing"),
        ("Filename labels", "20.22%", "Flat at 1.78"),
        ("Random chance", "16.7%", "1.79"),
    ],
    (55, 40, 40),
)

pdf.ln(2)
pdf.txt(
    "Baseline (44.94%) exploits spurious correlations in noisy data. Filename-label model "
    "is near random -- neither label source is trustworthy. With proper cleaning, expected "
    "60-70% accuracy (+15-25pp), driven entirely by data quality improvements."
)

out = BASE / "reports" / "Round1_Data_Detective_Report.pdf"
pdf.output(str(out))
print(f"Saved: {out}  ({out.stat().st_size / 1024:.0f} KB)")
