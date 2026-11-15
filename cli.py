"""
Command-Line Interface (CLI) for Zero-Blur Gemini & App Watermark Remover.
"""

import os
import sys
import argparse
import cv2
from watermark_core import (
    clean_all_watermarks,
    process_video_clean
)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Blur Gemini & App Watermark Remover (Mathematical De-Blend & Edge Preservation)"
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input video or image")
    parser.add_argument("-o", "--output", required=True, help="Path to output cleaned file")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        required=False,
        default=[540, 1100, 130, 130],
        metavar=("X", "Y", "W", "H"),
        help="Watermark box: X Y Width Height (default: 540 1100 130 130)"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "translucent", "solid_text"],
        default="auto",
        help="Removal mode: 'auto' (Recommended), 'translucent' (Gemini Sparkle), 'solid_text' (App Logos)"
    )
    parser.add_argument(
        "--sensitivity",
        type=int,
        default=60,
        help="Watermark sensitivity (10 to 90, default: 60)"
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    ext = os.path.splitext(input_path)[1].lower()

    boxes = [tuple(args.bbox)]

    if ext in image_exts:
        print(f"[*] Processing Image: {input_path}")
        img = cv2.imread(input_path)
        if img is None:
            print("Error: Could not read image.")
            sys.exit(1)

        cleaned = clean_all_watermarks(img, boxes, mode=args.mode, sensitivity=args.sensitivity)
        cv2.imwrite(output_path, cleaned)
        print(f"[+] Saved clean image (0% Blur / 100% Sharp) to: {output_path}")

    elif ext in video_exts:
        print(f"[*] Processing Video: {input_path}")

        def on_progress(current, total):
            percent = (current / total) * 100
            sys.stdout.write(f"\rProgress: [{current}/{total}] {percent:.1f}%")
            sys.stdout.flush()

        print(f"[*] Target box: {boxes[0]}")
        print(f"[*] Mode: {args.mode.upper()} | Rendering video with Zero Blur...")

        success = process_video_clean(
            input_video_path=input_path,
            output_video_path=output_path,
            boxes=boxes,
            mode=args.mode,
            sensitivity=args.sensitivity,
            progress_callback=on_progress
        )

        print()
        if success and os.path.exists(output_path):
            print(f"[+] Done! Saved clean video (audio intact, 0% blur) to: {output_path}")
        else:
            print("[-] Failed to process video.")
            sys.exit(1)
    else:
        print(f"Error: Unsupported file format '{ext}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
