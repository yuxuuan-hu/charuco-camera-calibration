# ChArUco Camera Calibration

A small OpenCV toolkit for generating a printable ChArUco board, capturing
calibration images from a USB/V4L2 camera, and estimating camera intrinsics from
the saved frames.

The calibration command writes two reports:

- `opencv_pinhole_rational`: OpenCV pinhole camera model with
  `CALIB_RATIONAL_MODEL`.
- `opencv_fisheye`: OpenCV fisheye model. For very wide lenses this may be
  useful, but check the reprojection error before using it.

## Environment

Python 3.10 or newer is recommended. The capture tool uses OpenCV GUI windows,
so install the normal `opencv-contrib-python` package, not the headless build.

```bash
git clone https://github.com/yuxuuan-hu/charuco-camera-calibration
cd charuco-camera-calibration

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you use conda:

```bash
conda env create -f environment.yml
conda activate charuco-calib
```

## 1. Generate And Print The Board

Generate an A4 PDF and PNG:

```bash
charuco-generate-board \
  --out-dir outputs/charuco_a4 \
  --squares-x 7 \
  --squares-y 10 \
  --square-mm 24 \
  --marker-mm 18 \
  --dictionary DICT_5X5_100
```

Print the PDF at 100 percent scale. Disable "fit to page" or similar printer
scaling options. After printing, measure one square and pass the measured square
size to the capture/calibration commands with `--square-mm`.

## 2. Capture Calibration Images

Move the board through the whole field of view: center, corners, edges, close,
far, and tilted angles. Save frames only when many ChArUco corners are detected.
For most cameras, 25 to 60 good images is a practical starting point.

Note: For the board printed on A4 paper, the square size is currently 23.1 mm and the marker size is 17.35 mm.

```bash
charuco-capture-images \
  --device /dev/video2 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --out-dir calibration_images/wrist \
  --prefix wrist_charuco \
  --squares-x 7 \
  --squares-y 10 \
  --square-mm 23.1 \
  --marker-mm 17.35
```

Preview keys:

| Key | Action |
| --- | --- |
| `SPACE` or `s` | Save the current raw frame. |
| `d` | Toggle ChArUco detection overlay. |
| `p` | Pause or resume preview. |
| `q` or `ESC` | Quit. |

Captured frames are saved as PNG files. A session JSON file is written beside
the images with the board settings and per-frame detection counts.

## 3. Calibrate

```bash
charuco-calibrate-camera \
  --images 'calibration_images/wrist/*.png' \
  --out-dir outputs/wrist_charuco \
  --squares-x 7 \
  --squares-y 10 \
  --square-mm 23.1 \
  --marker-mm 17.35 \
  --save-debug
```

Important outputs:

- `wrist_pinhole_rational_calibration.json`: pinhole+rational calibration.
- `wrist_fisheye_calibration.json`: fisheye calibration.
- `undistort_example_pinhole_rational.png`: sample undistorted image.
- `undistort_example.png`: sample fisheye undistorted image.
- `debug/`: optional detection overlay images when `--save-debug` is used.

Each JSON report contains image size, board settings, RMS, reprojection error,
camera matrix, distortion coefficients, and per-image detection metadata.

## Tips For Good Results

- Keep all captured images at the same resolution.
- Cover the full image area, especially the corners.
- Include strong board tilts so the solver sees perspective changes.
- Avoid motion blur and severe glare.
- Reject images with very few detected corners.
- Prefer the model with a low reprojection error and visually good undistortion.

## Development

Run basic import and bytecode checks:

```bash
python -m compileall src
charuco-generate-board --help
charuco-capture-images --help
charuco-calibrate-camera --help
```
