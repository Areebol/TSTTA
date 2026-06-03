#!/usr/bin/env python3
"""Compose 2 single-page PDFs into a vertical grid PDF.

功能：
- 读取每个 PDF 的第 1 页
- 单列自上而下拼在同一页里
- 子图下方使用固定图注

依赖：
- PyMuPDF (pip 包名：pymupdf)

示例：
    python compose_grid.py --dir ./vis_results/ablation_n --out combined_ablation_n.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

# =========================
# Tunable defaults (edit here)
# =========================
DEFAULT_ZOOM = 2.0
DEFAULT_MARGIN = 24
DEFAULT_GAP = 18

DEFAULT_LABEL_FONTSIZE = 36.0

DEFAULT_LEGEND_FONTSIZE = 50.0


def _require_fitz():
    try:
        import fitz  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少依赖 PyMuPDF：请先安装 `pip install pymupdf`。"
            f" 原始错误: {e}"
        )
    return fitz


def render_first_page_pixmap(pdf_path: Path, zoom: float):
    fitz = _require_fitz()
    doc = fitz.open(str(pdf_path))
    try:
        if doc.page_count < 1:
            raise ValueError(f"Empty PDF: {pdf_path}")
        page = doc.load_page(0)

        # 坐标轴去除逻辑 (保持原样)
        if getattr(render_first_page_pixmap, "_strip_axes", False):
            _strip_axes_text(
                page,
                left_frac=getattr(render_first_page_pixmap, "_strip_left_frac", 0.16),
                bottom_frac=getattr(render_first_page_pixmap, "_strip_bottom_frac", 0.16),
                padding=getattr(render_first_page_pixmap, "_strip_padding", 1.5),
            )

        mat = fitz.Matrix(zoom, zoom)
        return page.get_pixmap(matrix=mat, alpha=False)
    finally:
        doc.close()


def _strip_axes_text(page, left_frac: float, bottom_frac: float, padding: float) -> None:
    """Hide likely axis tick/label text by redacting text spans near left/bottom margins."""
    fitz = _require_fitz()
    rect = page.rect
    w = float(rect.width)
    h = float(rect.height)
    left_limit = rect.x0 + w * max(0.0, min(0.5, float(left_frac)))
    bottom_limit = rect.y1 - h * max(0.0, min(0.5, float(bottom_frac)))

    text_dict = page.get_text("dict")
    redact_rects: list[fitz.Rect] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = span.get("bbox")
                if not bbox:
                    continue
                x0, y0, x1, y1 = bbox
                in_left_strip = x1 <= left_limit
                in_bottom_strip = y0 >= bottom_limit
                if not (in_left_strip or in_bottom_strip):
                    continue
                r = fitz.Rect(x0, y0, x1, y1)
                if padding:
                    r = r + (-padding, -padding, padding, padding)
                redact_rects.append(r)

    if not redact_rects:
        return

    for r in redact_rects:
        page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)


def resolve_input_pdfs(dir_path: Path, files: list[str] | None, out_path: Path) -> list[Path]:
    if files:
        pdfs = [Path(f).expanduser().resolve() for f in files]
    else:
        pdfs = sorted(dir_path.glob("*.pdf"))

    out_name = out_path.name
    pdfs = [
        p
        for p in pdfs
        if p.suffix.lower() == ".pdf" and p.exists() and p.name != out_name
    ]
    if len(pdfs) != 2:
        raise ValueError(
            f"Expected exactly 2 PDFs, got {len(pdfs)}: {[p.name for p in pdfs]}. "
        )
    return pdfs


def format_label_from_filename(pdf_path: Path) -> str:
    stem = pdf_path.stem
    if "_2_" in stem:
        left, right = stem.split("_2_", 1)
        return f"{left} -> {right}"
    return stem


def main() -> None:
    fitz = _require_fitz()

    parser = argparse.ArgumentParser()
    default_dir = './vis_results/ablation_n'
    parser.add_argument("--dir", default=default_dir)
    parser.add_argument("--out", default="combined_ablation_n.pdf")
    parser.add_argument("--files", nargs="*", default=None)
    parser.add_argument("--zoom", type=float, default=DEFAULT_ZOOM)
    # --label-fontsize controls caption font size
    parser.add_argument("--label-fontsize", type=float, default=DEFAULT_LABEL_FONTSIZE)

    # Axes stripping
    parser.add_argument("--keep-axes", dest="strip_axes", action="store_false")
    parser.add_argument("--strip-left-frac", type=float, default=0.16)
    parser.add_argument("--strip-bottom-frac", type=float, default=0.16)
    parser.add_argument("--strip-padding", type=float, default=1.5)

    parser.set_defaults(strip_axes=False)
    args = parser.parse_args()

    dir_path = Path(args.dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    pdfs = resolve_input_pdfs(dir_path, args.files, out_path=out_path)

    render_first_page_pixmap._strip_axes = bool(args.strip_axes)  # type: ignore
    render_first_page_pixmap._strip_left_frac = float(args.strip_left_frac)  # type: ignore
    render_first_page_pixmap._strip_bottom_frac = float(args.strip_bottom_frac)  # type: ignore
    render_first_page_pixmap._strip_padding = float(args.strip_padding)  # type: ignore

    print("Rendering pages...")
    pixmaps = [render_first_page_pixmap(p, zoom=args.zoom) for p in pdfs]

    # Layout parameters (points)
    margin = DEFAULT_MARGIN
    gap = DEFAULT_GAP
    label_fontsize = float(args.label_fontsize)
    label_h = max(40, int(label_fontsize * 1.4))
    
    # 获取最大的子图尺寸
    cell_w = max(p.width for p in pixmaps)
    cell_h = max(p.height for p in pixmaps)

    # 单行左右排列
    page_w = margin * 2 + cell_w * 2 + gap
    page_h = margin * 2 + cell_h + label_h

    doc_out = fitz.open()
    page = doc_out.new_page(width=page_w, height=page_h)

    # =========================================================
    # 单行拼图 (2 张左右排列)
    # =========================================================
    start_x = margin
    start_y = margin

    labels = ["ETTh1", "ETTh1->ETTh2"]

    for col, (pdf_path, pix) in enumerate(zip(pdfs, pixmaps)):
        # 计算坐标
        x0 = start_x + col * (cell_w + gap)
        y0 = start_y

        img_rect = fitz.Rect(x0, y0, x0 + cell_w, y0 + cell_h)
        page.insert_image(img_rect, stream=pix.tobytes("png"), keep_proportion=True)

        # 手动计算居中位置以避免 insert_textbox 因空间不够而截断隐藏的问题
        label_text = labels[col] if col < len(labels) else format_label_from_filename(pdf_path)
        text_length = fitz.get_text_length(label_text, fontsize=label_fontsize)
        text_x = x0 + (cell_w - text_length) / 2
        text_y = y0 + cell_h + label_h * 0.7  # 垂直方向微调
        
        page.insert_text(
            (text_x, text_y),
            label_text,
            fontsize=label_fontsize,
            color=(0, 0, 0),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc_out.save(str(out_path))
    doc_out.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
