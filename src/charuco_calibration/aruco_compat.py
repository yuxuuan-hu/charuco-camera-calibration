"""Compatibility helpers for OpenCV's evolving cv2.aruco API."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def require_aruco() -> Any:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not include cv2.aruco. "
            "Install opencv-contrib-python instead of opencv-python-headless."
        )
    return cv2.aruco


def get_dictionary(name: str) -> Any:
    aruco = require_aruco()
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    dictionary_id = getattr(aruco, name)
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dictionary_id)
    return aruco.Dictionary_get(dictionary_id)


def create_charuco_board(
    squares_x: int,
    squares_y: int,
    square_length: float,
    marker_length: float,
    dictionary: Any,
) -> Any:
    aruco = require_aruco()
    if hasattr(aruco, "CharucoBoard_create"):
        return aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            square_length,
            marker_length,
            dictionary,
        )
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length,
            marker_length,
            dictionary,
        )
    raise RuntimeError("This OpenCV build does not provide ChArUco board support.")


def create_detector_parameters() -> Any:
    aruco = require_aruco()
    if hasattr(aruco, "DetectorParameters_create"):
        return aruco.DetectorParameters_create()
    return aruco.DetectorParameters()


def draw_charuco_board(board: Any, image_size: tuple[int, int]) -> np.ndarray:
    if hasattr(board, "draw"):
        return board.draw(image_size, marginSize=0, borderBits=1)
    if hasattr(board, "generateImage"):
        return board.generateImage(image_size, marginSize=0, borderBits=1)
    raise RuntimeError("This OpenCV ChArUco board cannot render itself.")


def chessboard_corners(board: Any) -> np.ndarray:
    if hasattr(board, "chessboardCorners"):
        return np.asarray(board.chessboardCorners, dtype=np.float64)
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    raise RuntimeError("This OpenCV ChArUco board does not expose chessboard corners.")
