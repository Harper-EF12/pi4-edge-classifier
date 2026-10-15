#!/usr/bin/env python3
"""Capture training images from a USB webcam, one folder per class.

Usage:
    python capture_images.py keyboard mouse background

A preview window opens. Each class gets a number key:
    1, 2, 3 ...  save an image to that class (hold the key to save a burst)
    q or Esc     finish, and pack everything into my_images.zip for Colab

Images are saved to my_images/<class>/. Running the script again adds to
the existing folders, so you can collect images over several sessions.
Only the area inside the green square is saved: it is the same centre crop
that classify.py uses, so training and live images match.
"""
import argparse
import os
import re
import shutil
import time

import cv2

SAVE_SIZE = 320   # saved images are 320 x 320; the notebook resizes them to 224
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def centre_square(frame):
    """Return the largest centred square of the frame, and its (x, y, side)."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x, y = (w - side) // 2, (h - side) // 2
    return frame[y:y + side, x:x + side], (x, y, side)


def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(f.lower().endswith(IMAGE_EXT) for f in os.listdir(folder))


def make_zip(out_dir):
    """Zip out_dir as <out_dir>.zip, keeping the top folder name inside the zip."""
    parent = os.path.dirname(os.path.abspath(out_dir))
    name = os.path.basename(os.path.abspath(out_dir))
    return shutil.make_archive(os.path.join(parent, name), "zip", root_dir=parent, base_dir=name)


def draw_overlay(frame, box, classes, counts, flash_class):
    x, y, side = box
    colour = (0, 255, 255) if flash_class is not None else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + side - 1, y + side - 1), colour, 3)

    lines = [f"{i + 1}: {c}  ({counts[c]})" for i, c in enumerate(classes)]
    lines.append("q: finish and zip")
    height = 26 * len(lines) + 12
    cv2.rectangle(frame, (0, 0), (260, height), (0, 0, 0), -1)
    for i, text in enumerate(lines):
        is_flash = i < len(classes) and classes[i] == flash_class
        cv2.putText(frame, text, (10, 26 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 255) if is_flash else (255, 255, 255), 2 if is_flash else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("classes", nargs="+", help="class names, e.g. keyboard mouse background")
    ap.add_argument("--out", default="my_images", help="output folder (default: my_images)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--interval", type=float, default=0.25,
                    help="minimum seconds between saves while a key is held (default 0.25)")
    args = ap.parse_args()

    classes = args.classes
    if len(classes) < 2:
        raise SystemExit("Give at least two class names.")
    if len(classes) > 9:
        raise SystemExit("At most 9 classes (keys 1 to 9).")
    if len(set(classes)) != len(classes):
        raise SystemExit("Class names must be different.")
    for c in classes:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", c):
            raise SystemExit(f"Bad class name '{c}': use letters, numbers, - or _ only (no spaces).")

    for c in classes:
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
    counts = {c: count_images(os.path.join(args.out, c)) for c in classes}

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Check `ls /dev/video*`.")

    print("Press 1-9 to save images (hold for a burst), q to finish.")
    last_save, flash_class, flash_until = 0.0, None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(0.5)
                continue

            square, box = centre_square(frame)
            now = time.monotonic()
            if now > flash_until:
                flash_class = None

            preview = frame.copy()
            draw_overlay(preview, box, classes, counts, flash_class)
            cv2.imshow("Capture images (q to finish)", preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if ord("1") <= key < ord("1") + len(classes) and now - last_save >= args.interval:
                c = classes[key - ord("1")]
                img = cv2.resize(square, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
                name = f"{c}_{time.strftime('%Y%m%d_%H%M%S')}_{counts[c]:04d}.jpg"
                cv2.imwrite(os.path.join(args.out, c, name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                counts[c] += 1
                last_save, flash_class, flash_until = now, c, now + 0.15
                print(f"saved {c:>15}: {counts[c]}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("\nImages per class:")
    for c in classes:
        note = "  <- aim for at least 100" if counts[c] < 100 else ""
        print(f"{c:>15}: {counts[c]}{note}")

    if sum(counts.values()) == 0:
        print("No images saved, so no zip created.")
        return
    zip_path = make_zip(args.out)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"\nCreated {zip_path} ({size_mb:.1f} MB).")
    print("Upload this file in the notebook's 'Option B: your own images' cell.")


if __name__ == "__main__":
    main()
