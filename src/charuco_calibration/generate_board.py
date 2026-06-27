"""Generate an A4 printable ChArUco board."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .aruco_compat import create_charuco_board, draw_charuco_board, get_dictionary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an A4 ChArUco board as PNG and PDF.")
    parser.add_argument("--out-dir", default="outputs/charuco_a4")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-mm", type=float, default=24.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    return parser.parse_args()


def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    a4_width_mm = 210.0
    a4_height_mm = 297.0
    page_width_px = mm_to_px(a4_width_mm, args.dpi)
    page_height_px = mm_to_px(a4_height_mm, args.dpi)
    square_px = mm_to_px(args.square_mm, args.dpi)
    board_width_px = square_px * args.squares_x
    board_height_px = square_px * args.squares_y

    if board_width_px > page_width_px or board_height_px > page_height_px:
        raise ValueError("Board does not fit on A4 with the requested square size.")

    dictionary = get_dictionary(args.dictionary)
    board = create_charuco_board(
        args.squares_x,
        args.squares_y,
        float(args.square_mm),
        float(args.marker_mm),
        dictionary,
    )
    board_image = draw_charuco_board(board, (board_width_px, board_height_px))

    page = np.full((page_height_px, page_width_px), 255, dtype=np.uint8)
    x0 = (page_width_px - board_width_px) // 2
    y0 = (page_height_px - board_height_px) // 2
    page[y0 : y0 + board_height_px, x0 : x0 + board_width_px] = board_image

    stem = (
        f"charuco_a4_{args.squares_x}x{args.squares_y}_"
        f"square{args.square_mm:g}mm_marker{args.marker_mm:g}mm_{args.dictionary}"
    )
    png_path = out_dir / f"{stem}_{args.dpi}dpi.png"
    pdf_path = out_dir / f"{stem}.pdf"
    json_path = out_dir / f"{stem}.json"

    pil_image = Image.fromarray(page)
    pil_image.save(png_path, dpi=(args.dpi, args.dpi))
    pil_image.convert("RGB").save(pdf_path, "PDF", resolution=args.dpi)

    metadata = {
        "dictionary": args.dictionary,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_mm": args.square_mm,
        "marker_length_mm": args.marker_mm,
        "dpi": args.dpi,
        "a4_size_mm": [a4_width_mm, a4_height_mm],
        "page_size_px": [page_width_px, page_height_px],
        "board_size_px": [board_width_px, board_height_px],
        "board_origin_px": [x0, y0],
        "notes": [
            "Print at 100% scale, actual size, no fit-to-page.",
            "After printing, measure one square and use the measured size for calibration.",
        ],
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Generated ChArUco A4 board")
    print(f"  PNG:      {png_path}")
    print(f"  PDF:      {pdf_path}")
    print(f"  metadata: {json_path}")
    print(f"  squares:  {args.squares_x} x {args.squares_y}")
    print(f"  square:   {args.square_mm:g} mm")
    print(f"  marker:   {args.marker_mm:g} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
