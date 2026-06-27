"""Interactively capture raw ChArUco calibration images from a camera."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .aruco_compat import create_charuco_board, create_detector_parameters, get_dictionary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a camera and save raw frames for ChArUco calibration."
    )
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--buffer-size", type=int, default=1)
    parser.add_argument(
        "--out-dir",
        default="calibration_images",
        help="Directory where captured PNG frames will be saved.",
    )
    parser.add_argument("--prefix", default="charuco")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-mm", type=float, default=24.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--no-detect", action="store_true", help="Disable live ChArUco overlay.")
    parser.add_argument("--window-name", default="ChArUco capture")
    return parser.parse_args()


def make_board(args: argparse.Namespace):
    dictionary = get_dictionary(args.dictionary)
    board = create_charuco_board(
        args.squares_x,
        args.squares_y,
        float(args.square_mm),
        float(args.marker_mm),
        dictionary,
    )
    return board, dictionary


def video_source(device: str) -> str | int:
    return int(device) if device.isdigit() else device


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_source(args.device))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera device: {args.device}")
    if args.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffer_size)
    return cap


def detect_charuco(
    frame_bgr: np.ndarray,
    board,
    dictionary,
) -> tuple[np.ndarray, int, int]:
    debug = frame_bgr.copy()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    params = create_detector_parameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    marker_count = 0 if marker_ids is None else int(len(marker_ids))
    corner_count = 0

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners,
            marker_ids,
            gray,
            board,
        )
        if retval is not None and retval > 0 and charuco_ids is not None:
            corner_count = int(len(charuco_ids))
            cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
    return debug, marker_count, corner_count


def overlay_status(
    frame_bgr: np.ndarray,
    saved_count: int,
    marker_count: int | None,
    corner_count: int | None,
    min_corners: int,
    paused: bool,
) -> np.ndarray:
    out = frame_bgr.copy()
    if corner_count is None:
        quality = "detect off"
        color = (0, 255, 255)
    else:
        ok = corner_count >= min_corners
        quality = f"markers {marker_count} corners {corner_count}"
        color = (0, 220, 0) if ok else (0, 0, 255)

    lines = [
        f"{quality} | saved {saved_count}",
        "SPACE/s save | d detect | p pause | q quit",
    ]
    if paused:
        lines.append("PAUSED")

    y = 24
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 26
    return out


def save_frame(
    frame_bgr: np.ndarray,
    out_dir: Path,
    prefix: str,
    index: int,
    marker_count: int | None,
    corner_count: int | None,
) -> tuple[Path, dict[str, object]]:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{prefix}_{timestamp}_{index:03d}.png"
    if not cv2.imwrite(str(path), frame_bgr):
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    record = {
        "image": str(path),
        "timestamp": timestamp,
        "index": index,
        "markers": marker_count,
        "charuco_corners": corner_count,
    }
    return path, record


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_path = out_dir / f"{args.prefix}_capture_session_{time.strftime('%Y%m%d_%H%M%S')}.json"

    board = None
    dictionary = None
    detect_enabled = not args.no_detect
    if detect_enabled:
        board, dictionary = make_board(args)

    cap = open_camera(args)
    saved_records: list[dict[str, object]] = []
    saved_count = 0
    paused = False
    last_frame: np.ndarray | None = None
    last_marker_count: int | None = None
    last_corner_count: int | None = None

    print("Interactive ChArUco capture")
    print(f"  device: {args.device}")
    print(f"  request: {args.width}x{args.height} {args.fps:g}fps {args.fourcc}")
    print(f"  out_dir: {out_dir}")
    print("  keys: SPACE/s save, d toggle detection, p pause, q/ESC quit")

    try:
        while True:
            if not paused or last_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read a frame from {args.device}")
                last_frame = frame

                if detect_enabled and board is not None and dictionary is not None:
                    preview, last_marker_count, last_corner_count = detect_charuco(
                        frame, board, dictionary
                    )
                else:
                    preview = frame
                    last_marker_count = None
                    last_corner_count = None
            else:
                frame = last_frame
                preview = frame
                if detect_enabled and board is not None and dictionary is not None:
                    preview, last_marker_count, last_corner_count = detect_charuco(
                        frame, board, dictionary
                    )

            shown = overlay_status(
                preview,
                saved_count,
                last_marker_count,
                last_corner_count,
                args.min_corners,
                paused,
            )
            cv2.imshow(args.window_name, shown)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            if key in (ord("d"), ord("D")):
                detect_enabled = not detect_enabled
                if detect_enabled and (board is None or dictionary is None):
                    board, dictionary = make_board(args)
                print(f"Detection overlay: {'on' if detect_enabled else 'off'}")
            elif key in (ord("p"), ord("P")):
                paused = not paused
                print(f"Preview: {'paused' if paused else 'running'}")
            elif key in (32, ord("s"), ord("S")):
                saved_count += 1
                path, record = save_frame(
                    last_frame,
                    out_dir,
                    args.prefix,
                    saved_count,
                    last_marker_count,
                    last_corner_count,
                )
                saved_records.append(record)
                session = {
                    "device": args.device,
                    "requested_width": args.width,
                    "requested_height": args.height,
                    "requested_fps": args.fps,
                    "fourcc": args.fourcc,
                    "dictionary": args.dictionary,
                    "squares_x": args.squares_x,
                    "squares_y": args.squares_y,
                    "square_length_mm": args.square_mm,
                    "marker_length_mm": args.marker_mm,
                    "min_corners": args.min_corners,
                    "captures": saved_records,
                }
                session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
                print(
                    f"Saved {path} "
                    f"(markers={last_marker_count}, corners={last_corner_count})"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved_count} frame(s)")
    print(f"Session metadata: {session_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
