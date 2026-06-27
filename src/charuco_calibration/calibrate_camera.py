"""Calibrate a camera from saved ChArUco images."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

from .aruco_compat import (
    chessboard_corners,
    create_charuco_board,
    create_detector_parameters,
    get_dictionary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate a camera from ChArUco images.")
    parser.add_argument("--images", required=True, help="Glob for images, e.g. 'calibration_images/*.png'.")
    parser.add_argument("--out-dir", default="outputs/charuco_calibration")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=10)
    parser.add_argument("--square-mm", type=float, default=24.0)
    parser.add_argument("--marker-mm", type=float, default=18.0)
    parser.add_argument("--dictionary", default="DICT_5X5_100")
    parser.add_argument("--min-corners", type=int, default=12)
    parser.add_argument("--save-debug", action="store_true")
    return parser.parse_args()


def detect_charuco(
    image_bgr: np.ndarray,
    board,
    dictionary,
) -> tuple[np.ndarray | None, np.ndarray | None, list[np.ndarray], np.ndarray | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    params = create_detector_parameters()
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if marker_ids is None or len(marker_ids) == 0:
        return None, None, marker_corners, marker_ids
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    if retval is None or retval <= 0:
        return None, None, marker_corners, marker_ids
    return charuco_corners, charuco_ids, marker_corners, marker_ids


def charuco_object_points(board, charuco_ids: np.ndarray) -> np.ndarray:
    corners = chessboard_corners(board)
    ids = charuco_ids.reshape(-1)
    return corners[ids].reshape(-1, 1, 3)


def compute_reprojection_error(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    k_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_error_sq = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, k_matrix, distortion)
        delta = projected.reshape(-1, 2) - img.reshape(-1, 2)
        total_error_sq += float((delta * delta).sum())
        total_points += int(img.shape[0])
    if total_points == 0:
        return float("nan")
    return float(np.sqrt(total_error_sq / total_points))


def compute_pinhole_reprojection_error(
    board,
    all_corners: list[np.ndarray],
    all_ids: list[np.ndarray],
    rvecs,
    tvecs,
    k_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_error_sq = 0.0
    total_points = 0
    board_corners = chessboard_corners(board)
    for corners, ids, rvec, tvec in zip(all_corners, all_ids, rvecs, tvecs):
        obj = board_corners[ids.reshape(-1)].reshape(-1, 1, 3)
        projected, _ = cv2.projectPoints(obj, rvec, tvec, k_matrix, distortion)
        delta = projected.reshape(-1, 2) - corners.reshape(-1, 2)
        total_error_sq += float((delta * delta).sum())
        total_points += int(corners.shape[0])
    if total_points == 0:
        return float("nan")
    return float(np.sqrt(total_error_sq / total_points))


def first_readable_image(image_paths: list[Path]) -> np.ndarray:
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    raise RuntimeError("No readable calibration images.")


def main() -> int:
    args = parse_args()
    image_paths = [Path(path) for path in sorted(glob.glob(args.images))]
    if not image_paths:
        raise RuntimeError(f"No images matched: {args.images}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_dictionary(args.dictionary)
    board = create_charuco_board(
        args.squares_x,
        args.squares_y,
        float(args.square_mm),
        float(args.marker_mm),
        dictionary,
    )

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    charuco_corners_all: list[np.ndarray] = []
    charuco_ids_all: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    per_image: list[dict[str, object]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            per_image.append({"image": str(image_path), "used": False, "reason": "read_failed"})
            continue
        height, width = image.shape[:2]
        if image_size is None:
            image_size = (width, height)
        elif image_size != (width, height):
            per_image.append({"image": str(image_path), "used": False, "reason": "size_mismatch"})
            continue

        charuco_corners, charuco_ids, marker_corners, marker_ids = detect_charuco(
            image, board, dictionary
        )
        corner_count = 0 if charuco_ids is None else int(len(charuco_ids))
        marker_count = 0 if marker_ids is None else int(len(marker_ids))
        used = charuco_corners is not None and charuco_ids is not None and corner_count >= args.min_corners
        per_image.append(
            {
                "image": str(image_path),
                "used": used,
                "charuco_corners": corner_count,
                "markers": marker_count,
            }
        )

        if args.save_debug:
            debug = image.copy()
            if marker_ids is not None and len(marker_ids) > 0:
                cv2.aruco.drawDetectedMarkers(debug, marker_corners, marker_ids)
            if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
                cv2.aruco.drawDetectedCornersCharuco(debug, charuco_corners, charuco_ids)
            cv2.imwrite(str(debug_dir / image_path.name), debug)

        if not used:
            continue

        object_points.append(charuco_object_points(board, charuco_ids))
        image_points.append(np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 1, 2))
        charuco_corners_all.append(np.asarray(charuco_corners, dtype=np.float32))
        charuco_ids_all.append(np.asarray(charuco_ids, dtype=np.int32))

    if image_size is None:
        raise RuntimeError("No readable calibration images.")
    if len(object_points) < 8:
        raise RuntimeError(f"Need at least 8 usable images, got {len(object_points)}.")

    pinhole_k = np.zeros((3, 3), dtype=np.float64)
    pinhole_dist = np.zeros((14, 1), dtype=np.float64)
    pinhole_flags = cv2.CALIB_RATIONAL_MODEL
    pinhole_rms, pinhole_k, pinhole_dist, pinhole_rvecs, pinhole_tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=charuco_corners_all,
        charucoIds=charuco_ids_all,
        board=board,
        imageSize=image_size,
        cameraMatrix=pinhole_k,
        distCoeffs=pinhole_dist,
        flags=pinhole_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    pinhole_reprojection_error = compute_pinhole_reprojection_error(
        board,
        charuco_corners_all,
        charuco_ids_all,
        pinhole_rvecs,
        pinhole_tvecs,
        pinhole_k,
        pinhole_dist,
    )
    pinhole_report = {
        "camera_model": "opencv_pinhole_rational",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "dictionary": args.dictionary,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_mm": args.square_mm,
        "marker_length_mm": args.marker_mm,
        "usable_images": len(object_points),
        "total_images": len(image_paths),
        "rms": float(pinhole_rms),
        "reprojection_error_px": pinhole_reprojection_error,
        "camera_matrix": {"data": [float(v) for v in pinhole_k.reshape(-1)]},
        "distortion_model": "rational_polynomial",
        "distortion_coefficients": {"data": [float(v) for v in pinhole_dist.reshape(-1)]},
        "per_image": per_image,
    }
    pinhole_report_path = out_dir / "wrist_pinhole_rational_calibration.json"
    pinhole_report_path.write_text(json.dumps(pinhole_report, indent=2), encoding="utf-8")

    k_matrix = np.zeros((3, 3), dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in object_points]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        | cv2.fisheye.CALIB_CHECK_COND
        | cv2.fisheye.CALIB_FIX_SKEW
    )

    try:
        rms, k_matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            k_matrix,
            distortion,
            rvecs,
            tvecs,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )
    except cv2.error:
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
        rms, k_matrix, distortion, rvecs, tvecs = cv2.fisheye.calibrate(
            object_points,
            image_points,
            image_size,
            k_matrix,
            distortion,
            rvecs,
            tvecs,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        )

    reprojection_error = compute_reprojection_error(
        object_points,
        image_points,
        rvecs,
        tvecs,
        k_matrix,
        distortion,
    )
    report = {
        "camera_model": "opencv_fisheye",
        "image_width": image_size[0],
        "image_height": image_size[1],
        "dictionary": args.dictionary,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_mm": args.square_mm,
        "marker_length_mm": args.marker_mm,
        "usable_images": len(object_points),
        "total_images": len(image_paths),
        "rms": float(rms),
        "reprojection_error_px": reprojection_error,
        "camera_matrix": {"data": [float(v) for v in k_matrix.reshape(-1)]},
        "distortion_coefficients": {"data": [float(v) for v in distortion.reshape(-1)]},
        "per_image": per_image,
    }

    report_path = out_dir / "wrist_fisheye_calibration.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    example_image = first_readable_image(image_paths)
    pinhole_undistorted = cv2.undistort(example_image, pinhole_k, pinhole_dist)
    pinhole_undistorted_path = out_dir / "undistort_example_pinhole_rational.png"
    cv2.imwrite(str(pinhole_undistorted_path), pinhole_undistorted)

    new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        k_matrix,
        distortion,
        image_size,
        np.eye(3),
        balance=0.0,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        k_matrix,
        distortion,
        np.eye(3),
        new_k,
        image_size,
        cv2.CV_16SC2,
    )
    undistorted = cv2.remap(example_image, map1, map2, interpolation=cv2.INTER_LINEAR)
    undistorted_path = out_dir / "undistort_example.png"
    cv2.imwrite(str(undistorted_path), undistorted)

    print("ChArUco calibration complete")
    print(f"  usable images: {len(object_points)} / {len(image_paths)}")
    print("  pinhole+rational:")
    print(f"    rms:          {float(pinhole_rms):.4f}")
    print(f"    reproj error: {pinhole_reprojection_error:.4f} px")
    print(f"    report:       {pinhole_report_path}")
    print(f"    undistorted:  {pinhole_undistorted_path}")
    print("  fisheye:")
    print(f"    rms:          {float(rms):.4f}")
    print(f"    reproj error: {reprojection_error:.4f} px")
    print(f"    report:       {report_path}")
    print(f"    undistorted:  {undistorted_path}")
    if reprojection_error > 5.0 or np.allclose(distortion, 0.0):
        print("  warning: fisheye calibration did not converge; use the pinhole+rational result for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
