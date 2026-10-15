#!/usr/bin/env python3
"""Classify live USB webcam frames with a LiteRT (TensorFlow Lite) model.

Usage:
    python classify.py                         # int8 model, camera 0, preview window
    python classify.py --model model_float.tflite
    python classify.py --image test.jpg        # classify a single image file
    python classify.py --headless              # no window, print results only
    python classify.py --threshold 0.8         # say "unsure" below 80% confidence

Press q (or Esc) in the preview window to quit.
"""
import argparse
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def load_labels(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class Classifier:
    def __init__(self, model_path, threads=4):
        self.interpreter = Interpreter(model_path=model_path, num_threads=threads)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        _, self.height, self.width, _ = self.inp["shape"]

    def preprocess(self, frame_bgr):
        """Centre-crop to a square, resize, and convert BGR (OpenCV) to RGB (training)."""
        h, w = frame_bgr.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        square = frame_bgr[top:top + side, left:left + side]
        resized = cv2.resize(square, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def classify(self, frame_bgr):
        """Return (probabilities, inference time in ms)."""
        x = self.preprocess(frame_bgr)[np.newaxis].astype(np.float32)  # pixels 0..255
        if self.inp["dtype"] != np.float32:              # quantized model input
            scale, zero = self.inp["quantization"]
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(x / scale + zero), info.min, info.max)
        self.interpreter.set_tensor(self.inp["index"], x.astype(self.inp["dtype"]))

        start = time.perf_counter()
        self.interpreter.invoke()
        ms = (time.perf_counter() - start) * 1000

        probs = self.interpreter.get_tensor(self.out["index"])[0]
        if self.out["dtype"] != np.float32:              # quantized model output
            scale, zero = self.out["quantization"]
            probs = (probs.astype(np.float32) - zero) * scale
        return probs, ms


def describe(label, conf, threshold):
    """'cat 92%', or 'unsure (cat 55%)' when below the confidence threshold."""
    return f"{label} {conf:.0%}" if conf >= threshold else f"unsure ({label} {conf:.0%})"


def draw_overlay(frame, label, conf, ms, fps, threshold):
    sure = conf >= threshold
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(frame, describe(label, conf, threshold), (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255) if sure else (0, 165, 255), 2)
    cv2.putText(frame, f"inference {ms:.0f} ms | {fps:.1f} fps", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model_int8.tflite")
    ap.add_argument("--labels", default="labels.txt")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (/dev/videoN)")
    ap.add_argument("--image", help="classify this image file instead of the webcam")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--headless", action="store_true", help="no preview window")
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="report 'unsure' below this confidence, 0 to 1 (default 0.7)")
    args = ap.parse_args()

    labels = load_labels(args.labels)
    clf = Classifier(args.model, args.threads)
    if clf.out["shape"][-1] != len(labels):
        raise SystemExit(f"Model has {clf.out['shape'][-1]} outputs but {args.labels} has {len(labels)} labels")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        probs, ms = clf.classify(frame)
        for i in np.argsort(probs)[::-1]:
            print(f"{labels[i]:>15}: {probs[i]:.3f}")
        best = int(np.argmax(probs))
        print(f"Result: {describe(labels[best], probs[best], args.threshold)}  ({ms:.1f} ms)")
        return

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Check `ls /dev/video*`.")

    fps, last = 0.0, time.perf_counter()
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed, retrying...")
                time.sleep(0.5)
                continue

            probs, ms = clf.classify(frame)
            best = int(np.argmax(probs))
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - last, 1e-6))
            last = now

            if now - last_print > 0.5:
                print(f"{describe(labels[best], probs[best], args.threshold):>25}   ({ms:.0f} ms)")
                last_print = now

            if not args.headless:
                draw_overlay(frame, labels[best], probs[best], ms, fps, args.threshold)
                cv2.imshow("Classifier (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
