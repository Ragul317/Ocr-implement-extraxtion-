"""Render invoice-like test images (plus ground-truth text) for Phase 2 OCR testing.

These are synthetic but realistic documents rendered with real fonts; OCR on them
is real PaddleOCR. Real phone photos/scans should be added to tests/samples/ later.

Usage:
    python scripts/make_samples.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "tests" / "samples"

FONT_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

# Each entry: (text, font_path, size, x, gap_after). A row with "\t" is rendered as
# left text + right-aligned amount at the same y (like a real invoice line).
INVOICE_ROWS = [
    ("SHREE GANESH TRADERS", FONT_SANS_BOLD, 40, 60, 14),
    ("12, MG Road, Bengaluru 560001", FONT_SANS, 26, 60, 6),
    ("Phone: +91 98450 12345", FONT_SANS, 26, 60, 6),
    ("GSTIN: 29ABCDE1234F1Z5", FONT_SANS, 26, 60, 34),
    ("TAX INVOICE", FONT_SANS_BOLD, 34, 60, 24),
    ("Invoice No: INV-1001", FONT_SANS, 28, 60, 8),
    ("Date: 06/10/2026", FONT_SANS, 28, 60, 8),
    ("Customer: ABC Stores", FONT_SANS, 28, 60, 30),
    ("Item\tQty\tAmount", FONT_SANS_BOLD, 28, 60, 14),
    ("Milk 1L\t2\t₹50.00", FONT_SANS, 28, 60, 8),
    ("Bread\t1\t₹40.00", FONT_SANS, 28, 60, 8),
    ("Basmati Rice 5kg\t1\t₹500.00", FONT_SANS, 28, 60, 30),
    ("Subtotal\t\t₹590.00", FONT_SANS, 28, 60, 8),
    ("CGST 2.5%\t\t₹14.75", FONT_SANS, 28, 60, 8),
    ("SGST 2.5%\t\t₹14.75", FONT_SANS, 28, 60, 14),
    ("Total\t\t₹619.50", FONT_SANS_BOLD, 32, 60, 30),
    ("Thank you for your business!", FONT_SANS, 24, 60, 0),
]

RECEIPT_ROWS = [
    "   FRESH MART SUPERMARKET",
    "  Anna Nagar, Chennai 600040",
    "     Ph: 044-2615 7788",
    "--------------------------------",
    "Bill No: 58213   Date: 2026-09-28",
    "Time: 18:42      Cashier: R.Kumar",
    "--------------------------------",
    "Tomato 1kg            Rs. 36.00",
    "Onion 2kg             Rs. 64.00",
    "Toor Dal 1kg         Rs. 148.00",
    "Sunflower Oil 1L     Rs. 165.00",
    "--------------------------------",
    "Items: 4",
    "TOTAL                ₹ 413.00",
    "Paid (UPI)           ₹ 413.00",
    "--------------------------------",
    "    Thank you! Visit again",
]


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def render_invoice(width: int = 1240, height: int = 1300) -> tuple[Image.Image, list[str]]:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    truth: list[str] = []
    y = 60
    qty_x, amount_right = 760, width - 80
    for text, font_path, size, x, gap in INVOICE_ROWS:
        font = _font(font_path, size)
        if "\t" in text:
            left, qty, amount = text.split("\t")
            draw.text((x, y), left, font=font, fill="black")
            if qty:
                draw.text((qty_x, y), qty, font=font, fill="black")
            w = draw.textlength(amount, font=font)
            draw.text((amount_right - w, y), amount, font=font, fill="black")
            truth.append(" ".join(p for p in (left, qty, amount) if p))
        else:
            draw.text((x, y), text, font=font, fill="black")
            truth.append(text)
        y += size + gap
        if text.startswith(("GSTIN", "Customer", "Basmati")):
            draw.line((60, y - gap // 2, width - 60, y - gap // 2), fill="#888888", width=2)
    return img, truth


def render_receipt() -> tuple[Image.Image, list[str]]:
    font = _font(FONT_MONO, 24)
    line_h = 34
    img = Image.new("RGB", (520, 60 + line_h * len(RECEIPT_ROWS) + 40), "#f7f5ef")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(RECEIPT_ROWS):
        draw.text((20, 40 + i * line_h), text, font=font, fill="#222222")
    truth = [" ".join(t.split()) for t in RECEIPT_ROWS if t.strip("- ")]
    return img, truth


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    invoice, invoice_truth = render_invoice()
    receipt, receipt_truth = render_receipt()

    # Degraded copy: slight rotation, blur, JPEG compression (simulates a poor photo).
    blurry = invoice.rotate(1.5, expand=True, fillcolor="white").filter(ImageFilter.GaussianBlur(2.2))

    # Large copy: ~4x linear size (stresses downscaling / detection limits).
    large = invoice.resize((invoice.width * 4, invoice.height * 4), Image.Resampling.LANCZOS)

    blank = Image.new("RGB", (800, 600), "white")

    outputs = {
        "invoice_clear.png": (invoice, invoice_truth, {}),
        "receipt_mono.png": (receipt, receipt_truth, {}),
        "invoice_blurry.jpg": (blurry, invoice_truth, {"quality": 55}),
        "invoice_large.jpg": (large, invoice_truth, {"quality": 90}),
        "blank.png": (blank, [], {}),
    }
    for name, (image, truth, save_kwargs) in outputs.items():
        path = SAMPLES_DIR / name
        image.save(path, **save_kwargs)
        (SAMPLES_DIR / f"{path.stem}.truth.txt").write_text("\n".join(truth) + ("\n" if truth else ""), encoding="utf-8")
        print(f"{path.relative_to(SAMPLES_DIR.parent.parent)}  {image.width}x{image.height}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
