"""
Zero-Blur Watermark Remover - Desktop Application (Tkinter GUI)
Pure Python desktop tool - runs locally without any web browser.
Specifically engineered for Gemini AI Videos, App Logos, and Proper-Color Background Restoration.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

from watermark_core import (
    clean_watermark_zero_blur,
    clean_all_watermarks,
    process_video_clean,
    snap_to_watermark_peak,
    build_unified_video_masks
)


class GeminiWatermarkRemover(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gemini & App Watermark Remover (Proper Color & Zero Blur)")
        self.geometry("1300x890")
        self.minsize(1050, 720)

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # State variables
        self.current_filepath = None
        self.is_video = False
        self.cv_image = None              # Original current frame in BGR
        self.preview_cleaned_bgr = None   # Cleaned frame in BGR
        self.original_w = 0
        self.original_h = 0

        # Detected watermark boxes: [(x, y, w, h)]
        self.detected_boxes = []

        # Display scaling & offsets
        self.canvas_scale = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.displayed_w = 0
        self.displayed_h = 0

        # Mouse click/drag state
        self.drag_start_ix = None
        self.drag_start_iy = None
        self.drag_curr_cx = None
        self.drag_curr_cy = None

        # Box dimensions
        self.var_box_w = tk.IntVar(value=130)
        self.var_box_h = tk.IntVar(value=130)

        # Removal settings
        self.var_mode = tk.StringVar(value="auto")
        self.var_sensitivity = tk.IntVar(value=60)

        # View mode: 'boxes' or 'cleaned'
        self.current_view = "boxes"

        # Background processing flag
        self.is_processing = False
        self.cached_video_masks = None

        self._build_ui()

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # -----------------------------
        # LEFT CONTROL PANEL
        # -----------------------------
        sidebar = ttk.Frame(main_paned, width=430, padding=12)
        main_paned.add(sidebar, weight=0)

        # Header Title
        lbl_title = ttk.Label(
            sidebar,
            text="Watermark Remover",
            font=("Segoe UI", 16, "bold"),
            foreground="#0277BD"
        )
        lbl_title.pack(anchor="w", pady=(0, 2))

        lbl_sub = ttk.Label(
            sidebar,
            text="Proper Background Color Restoration & Zero-Blur Erasing",
            font=("Segoe UI", 9, "italic"),
            foreground="#555"
        )
        lbl_sub.pack(anchor="w", pady=(0, 10))

        # 1. Media File Section
        grp_file = ttk.LabelFrame(sidebar, text=" 1. Media File ", padding=10)
        grp_file.pack(fill=tk.X, pady=(0, 8))

        btn_open = ttk.Button(grp_file, text="📁 Open Video or Image File...", command=self.open_file)
        btn_open.pack(fill=tk.X, pady=(0, 5))

        self.lbl_file_info = ttk.Label(grp_file, text="No file loaded.", foreground="#555", wraplength=380)
        self.lbl_file_info.pack(anchor="w")

        # 2. Watermark Location Section
        grp_pos = ttk.LabelFrame(sidebar, text=" 2. Watermark Target (Auto-Snaps to Logo) ", padding=10)
        grp_pos.pack(fill=tk.X, pady=(0, 8))

        lbl_tap_hint = ttk.Label(
            grp_pos,
            text="👉 Click anywhere near the watermark! The box will automatically snap centered on the logo. Or drag a custom box.",
            font=("Segoe UI", 9, "bold"),
            foreground="#00897B",
            wraplength=380
        )
        lbl_tap_hint.pack(anchor="w", pady=(0, 6))

        # Quick preset buttons
        f_presets1 = ttk.Frame(grp_pos)
        f_presets1.pack(fill=tk.X, pady=2)
        btn_br = ttk.Button(f_presets1, text="↘️ Bottom-Right (Gemini)", command=lambda: self.apply_preset("bottom_right"))
        btn_br.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        btn_tr = ttk.Button(f_presets1, text="↗️ Top-Right", command=lambda: self.apply_preset("top_right"))
        btn_tr.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        f_presets2 = ttk.Frame(grp_pos)
        f_presets2.pack(fill=tk.X, pady=(2, 6))
        btn_bl = ttk.Button(f_presets2, text="↙️ Bottom-Left", command=lambda: self.apply_preset("bottom_left"))
        btn_bl.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        btn_tl = ttk.Button(f_presets2, text="↖️ Top-Left", command=lambda: self.apply_preset("top_left"))
        btn_tl.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # Box dimensions adjustment
        f_dims = ttk.Frame(grp_pos)
        f_dims.pack(fill=tk.X, pady=4)
        ttk.Label(f_dims, text="Box Width:").grid(row=0, column=0, sticky="w")
        sp_w = ttk.Spinbox(f_dims, from_=15, to=800, textvariable=self.var_box_w, width=7, command=self._on_size_changed)
        sp_w.grid(row=0, column=1, padx=4)
        sp_w.bind("<KeyRelease>", lambda e: self._on_size_changed())

        ttk.Label(f_dims, text="Box Height:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        sp_h = ttk.Spinbox(f_dims, from_=15, to=400, textvariable=self.var_box_h, width=7, command=self._on_size_changed)
        sp_h.grid(row=0, column=3, padx=4)
        sp_h.bind("<KeyRelease>", lambda e: self._on_size_changed())

        f_status_clear = ttk.Frame(grp_pos)
        f_status_clear.pack(fill=tk.X, pady=(5, 0))

        self.lbl_target_status = ttk.Label(
            f_status_clear,
            text="Tap on the watermark to target it.",
            font=("Segoe UI", 9, "bold"),
            foreground="#0277BD",
            wraplength=280
        )
        self.lbl_target_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_clear = ttk.Button(f_status_clear, text="❌ Clear", width=8, command=self.clear_boxes)
        btn_clear.pack(side=tk.RIGHT)

        # 3. Removal Engine Section (Proper Color & Zero Blur)
        grp_method = ttk.LabelFrame(sidebar, text=" 3. Removal Engine (Proper Color Guarantee) ", padding=10)
        grp_method.pack(fill=tk.X, pady=(0, 8))

        rb_auto = ttk.Radiobutton(
            grp_method,
            text="✨ Proper-Color Zero-Blur Engine (Recommended ★)",
            variable=self.var_mode,
            value="auto"
        )
        rb_auto.pack(anchor="w", pady=2)

        lbl_auto_desc = ttk.Label(
            grp_method,
            text="Removes 100% of the watermark and restores the exact true color of the background (e.g. black clothes become proper dark black, skin and nature colors 100% matched). Zero greyish haze, zero blur!",
            font=("Segoe UI", 8),
            foreground="#555",
            wraplength=380
        )
        lbl_auto_desc.pack(anchor="w", padx=20, pady=(0, 4))

        rb_trans = ttk.Radiobutton(
            grp_method,
            text="✨ Translucent Watermarks (Pure Alpha De-Blend)",
            variable=self.var_mode,
            value="translucent"
        )
        rb_trans.pack(anchor="w", pady=(2, 2))

        rb_solid = ttk.Radiobutton(
            grp_method,
            text="🔤 Solid Text & App Logos (Navier-Stokes Edge Preservation)",
            variable=self.var_mode,
            value="solid_text"
        )
        rb_solid.pack(anchor="w", pady=(2, 4))

        # Sensitivity slider
        f_sens = ttk.Frame(grp_method)
        f_sens.pack(fill=tk.X, padx=5, pady=(4, 6))
        self.lbl_sens_val = ttk.Label(f_sens, text=f"Sensitivity: {self.var_sensitivity.get()}%", width=16)
        self.lbl_sens_val.pack(side=tk.LEFT)
        sc_sens = ttk.Scale(
            f_sens, from_=10, to=90, variable=self.var_sensitivity, orient=tk.HORIZONTAL,
            command=self._on_sens_changed
        )
        sc_sens.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # 4. Actions Section
        grp_actions = ttk.LabelFrame(sidebar, text=" 4. Clean & Save ", padding=10)
        grp_actions.pack(fill=tk.X, pady=(0, 8))

        btn_preview = ttk.Button(
            grp_actions,
            text="👁️ Preview Cleaned Result (Verify Proper Color)",
            command=self.preview_cleaned_result
        )
        btn_preview.pack(fill=tk.X, pady=4)

        btn_save = ttk.Button(
            grp_actions,
            text="💾 Save Cleaned Video / Image...",
            command=self.save_cleaned_file
        )
        btn_save.pack(fill=tk.X, pady=4)

        # Progress bar & Status
        self.progress_bar = ttk.Progressbar(sidebar, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, pady=(10, 3))

        self.lbl_status = ttk.Label(sidebar, text="Ready.", foreground="#0277BD")
        self.lbl_status.pack(anchor="w")

        # -----------------------------
        # RIGHT PREVIEW CANVAS
        # -----------------------------
        right_frame = ttk.Frame(main_paned, padding=5)
        main_paned.add(right_frame, weight=1)

        # Top Toolbar
        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(toolbar, text="Display: ", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)

        ttk.Button(toolbar, text="Show Watermark Box (Green)", command=self.show_boxes_view).pack(side=tk.LEFT, padx=3)
        ttk.Button(toolbar, text="Show Cleaned Preview (Proper Color)", command=self.show_cleaned_view).pack(side=tk.LEFT, padx=3)

        self.lbl_canvas_legend = ttk.Label(
            toolbar,
            text="🎯 Click anywhere on the video to snap the watermark box | Drag to size custom box",
            foreground="#00897B",
            font=("Segoe UI", 9, "bold")
        )
        self.lbl_canvas_legend.pack(side=tk.RIGHT, padx=5)

        # Canvas
        self.canvas = tk.Canvas(right_frame, bg="#1E1E1E", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

    def _on_sens_changed(self, val):
        self.lbl_sens_val.config(text=f"Sensitivity: {int(float(val))}%")
        self.cached_video_masks = None

    def clear_boxes(self):
        self.detected_boxes = []
        self.cached_video_masks = None
        self.lbl_target_status.config(text="Boxes cleared. Tap on logo to target it.", foreground="#555")
        self.current_view = "boxes"
        self.redraw_canvas()

    # ----------------------------------------------------
    # FILE HANDLING
    # ----------------------------------------------------
    def open_file(self):
        filetypes = [
            ("Supported Media", "*.mp4;*.mkv;*.avi;*.mov;*.webm;*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
            ("Video Files", "*.mp4;*.mkv;*.avi;*.mov;*.webm"),
            ("Image Files", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"),
            ("All Files", "*.*")
        ]
        chosen = filedialog.askopenfilename(title="Select Video or Image File", filetypes=filetypes)
        if not chosen:
            return

        self.current_filepath = chosen
        ext = os.path.splitext(chosen)[1].lower()
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
        self.is_video = ext not in image_exts

        self.preview_cleaned_bgr = None

        if not self.is_video:
            img = cv2.imread(chosen)
            if img is None:
                messagebox.showerror("Error", f"Could not load image: {chosen}")
                return
            self.cv_image = img
            self.original_h, self.original_w = img.shape[:2]
            self.lbl_file_info.config(
                text=f"Image: {os.path.basename(chosen)}\nResolution: {self.original_w} x {self.original_h} px"
            )
        else:
            cap = cv2.VideoCapture(chosen)
            if not cap.isOpened():
                messagebox.showerror("Error", f"Could not open video: {chosen}")
                return
            self.original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                messagebox.showerror("Error", "Could not read frame from video.")
                return

            self.cv_image = frame
            self.lbl_file_info.config(
                text=f"Video: {os.path.basename(chosen)}\nResolution: {self.original_w} x {self.original_h} | {fps:.1f} FPS | {total_frames} Frames"
            )

        # Default size tuned for watermark constellation
        bw = max(130, int(self.original_w * 0.18))
        bh = max(130, int(self.original_h * 0.11))
        self.var_box_w.set(bw)
        self.var_box_h.set(bh)

        # Default to bottom right preset (accurately centered on Gemini logo)
        self.apply_preset("bottom_right")
        self.lbl_status.config(text="File loaded. Target box placed. Click 'Preview Cleaned Result' to verify.")

    def apply_preset(self, preset):
        if self.original_w == 0 or self.original_h == 0:
            return

        bw = self.var_box_w.get()
        bh = self.var_box_h.get()

        if preset == "bottom_right":
            # Gemini & app watermarks in portrait (720x1280) span around (605, 1165) -> 84% w, 91% h
            approx_x = int(self.original_w * 0.84)
            approx_y = int(self.original_h * 0.91)
            if self.cv_image is not None and not self.is_video:
                approx_x, approx_y = snap_to_watermark_peak(self.cv_image, approx_x, approx_y, search_radius=40)
            bx = max(0, min(approx_x - bw // 2, self.original_w - bw))
            by = max(0, min(approx_y - bh // 2, self.original_h - bh))
        elif preset == "top_right":
            approx_x = int(self.original_w * 0.889)
            approx_y = int(self.original_h * 0.078)
            bx = max(0, min(approx_x - bw // 2, self.original_w - bw))
            by = max(0, min(approx_y - bh // 2, self.original_h - bh))
        elif preset == "bottom_left":
            approx_x = int(self.original_w * 0.111)
            approx_y = int(self.original_h * 0.922)
            bx = max(0, min(approx_x - bw // 2, self.original_w - bw))
            by = max(0, min(approx_y - bh // 2, self.original_h - bh))
        elif preset == "top_left":
            approx_x = int(self.original_w * 0.111)
            approx_y = int(self.original_h * 0.078)
            bx = max(0, min(approx_x - bw // 2, self.original_w - bw))
            by = max(0, min(approx_y - bh // 2, self.original_h - bh))
        else:
            bx = max(0, self.original_w - bw - 20)
            by = max(0, self.original_h - bh - 20)

        self.detected_boxes = [(bx, by, bw, bh)]
        self.cached_video_masks = None
        self.lbl_target_status.config(
            text=f"Target centered at ({bx}, {by}) [{bw}x{bh}].",
            foreground="#00897B"
        )
        self.current_view = "boxes"
        self.redraw_canvas()

    def _on_size_changed(self):
        if not self.detected_boxes or self.original_w == 0:
            return
        bx, by, _, _ = self.detected_boxes[0]
        bw = self.var_box_w.get()
        bh = self.var_box_h.get()
        bx = max(0, min(bx, self.original_w - bw))
        by = max(0, min(by, self.original_h - bh))
        self.detected_boxes = [(bx, by, bw, bh)]
        self.cached_video_masks = None
        self.current_view = "boxes"
        self.redraw_canvas()

    # ----------------------------------------------------
    # MOUSE INTERACTION (TAP & DRAG)
    # ----------------------------------------------------
    def _canvas_to_img_coords(self, cx, cy):
        if self.canvas_scale <= 0:
            return (0, 0)
        ix = int((cx - self.canvas_offset_x) / self.canvas_scale)
        iy = int((cy - self.canvas_offset_y) / self.canvas_scale)
        ix = max(0, min(ix, self.original_w - 1))
        iy = max(0, min(iy, self.original_h - 1))
        return (ix, iy)

    def _img_to_canvas_coords(self, ix, iy):
        cx = int(ix * self.canvas_scale + self.canvas_offset_x)
        cy = int(iy * self.canvas_scale + self.canvas_offset_y)
        return (cx, cy)

    def _on_mouse_down(self, event):
        if self.cv_image is None:
            return
        if (event.x < self.canvas_offset_x or event.x > self.canvas_offset_x + self.displayed_w or
            event.y < self.canvas_offset_y or event.y > self.canvas_offset_y + self.displayed_h):
            return

        self.drag_start_ix, self.drag_start_iy = self._canvas_to_img_coords(event.x, event.y)
        self.drag_curr_cx = event.x
        self.drag_curr_cy = event.y

    def _on_mouse_drag(self, event):
        if self.drag_start_ix is None or self.cv_image is None:
            return
        self.drag_curr_cx = event.x
        self.drag_curr_cy = event.y

        self.redraw_canvas()
        c_x1, c_y1 = self._img_to_canvas_coords(self.drag_start_ix, self.drag_start_iy)
        self.canvas.create_rectangle(c_x1, c_y1, event.x, event.y, outline="#00E676", width=2, dash=(4, 2))

    def _on_mouse_up(self, event):
        if self.drag_start_ix is None or self.cv_image is None:
            return

        curr_ix, curr_iy = self._canvas_to_img_coords(event.x, event.y)
        dist = abs(curr_ix - self.drag_start_ix) + abs(curr_iy - self.drag_start_iy)

        bw = self.var_box_w.get()
        bh = self.var_box_h.get()

        if dist < 8:
            # 🎯 SINGLE CLICK / TAP: Auto-snap to exact watermark peak with generous constellation coverage!
            target_x, target_y = snap_to_watermark_peak(
                self.cv_image, self.drag_start_ix, self.drag_start_iy, search_radius=45
            )
            bw = max(130, self.var_box_w.get())
            bh = max(130, self.var_box_h.get())
            self.var_box_w.set(bw)
            self.var_box_h.set(bh)
            bx = max(0, min(target_x - bw // 2, self.original_w - bw))
            by = max(0, min(target_y - bh // 2, self.original_h - bh))
            self.detected_boxes = [(bx, by, bw, bh)]
            self.lbl_target_status.config(
                text=f"🎯 Snapped to logo! Centered at ({bx}, {by}) [{bw}x{bh}]. Ready to clean.",
                foreground="#00897B"
            )
        else:
            # ✏️ DRAGGED: Custom size box
            bx = min(self.drag_start_ix, curr_ix)
            by = min(self.drag_start_iy, curr_iy)
            bw = max(15, abs(curr_ix - self.drag_start_ix))
            bh = max(15, abs(curr_iy - self.drag_start_iy))
            self.detected_boxes = [(bx, by, bw, bh)]
            self.var_box_w.set(bw)
            self.var_box_h.set(bh)
            self.lbl_target_status.config(
                text=f"Custom box placed at ({bx}, {by}) [{bw}x{bh}].",
                foreground="#00897B"
            )

        self.cached_video_masks = None
        self.current_view = "boxes"
        self.drag_start_ix = None
        self.drag_start_iy = None
        self.redraw_canvas()

    def _on_canvas_resize(self, event):
        self.redraw_canvas()

    # ----------------------------------------------------
    # REMOVAL PREVIEW & RENDERING (PROPER COLOR)
    # ----------------------------------------------------
    def preview_cleaned_result(self):
        if self.cv_image is None:
            messagebox.showinfo("Info", "Please open a video or image file first.")
            return

        if not self.detected_boxes:
            self.apply_preset("bottom_right")

        self.lbl_status.config(text="Erasing watermark and restoring proper background color...")
        self.update_idletasks()

        precomputed = None
        if self.is_video and self.current_filepath and os.path.exists(self.current_filepath):
            # Precompute temporally invariant video mask so preview has 0% blur on hair/clothes!
            precomputed = build_unified_video_masks(
                self.current_filepath,
                self.detected_boxes,
                sensitivity=self.var_sensitivity.get()
            )
            self.cached_video_masks = precomputed

        cleaned = clean_all_watermarks(
            self.cv_image,
            self.detected_boxes,
            mode=self.var_mode.get(),
            sensitivity=self.var_sensitivity.get(),
            precomputed_masks=precomputed
        )

        self.preview_cleaned_bgr = cleaned
        self.current_view = "cleaned"
        self.redraw_canvas()
        self.lbl_status.config(text="Cleaned preview ready! Proper background color restored, zero blur.")

    def show_boxes_view(self):
        self.current_view = "boxes"
        self.lbl_canvas_legend.config(
            text="🎯 Target Mode: Green Box over Watermark | Click or drag on video to reposition",
            foreground="#00E676"
        )
        self.redraw_canvas()

    def show_cleaned_view(self):
        if self.preview_cleaned_bgr is None:
            self.preview_cleaned_result()
        else:
            self.current_view = "cleaned"
            self.lbl_canvas_legend.config(
                text="✨ Cleaned Preview (Zero Blur): Watermark erased, true background restored",
                foreground="#00E5FF"
            )
            self.redraw_canvas()

    def redraw_canvas(self):
        if self.cv_image is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 300,
                self.canvas.winfo_height() // 2 or 250,
                text="Click 'Open Video or Image File' to begin",
                fill="#888",
                font=("Segoe UI", 14)
            )
            return

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        base_img = self.preview_cleaned_bgr if (self.current_view == "cleaned" and self.preview_cleaned_bgr is not None) else self.cv_image

        h, w = base_img.shape[:2]
        scale_x = cw / w
        scale_y = ch / h
        self.canvas_scale = min(scale_x, scale_y) * 0.96

        self.displayed_w = int(w * self.canvas_scale)
        self.displayed_h = int(h * self.canvas_scale)
        self.canvas_offset_x = (cw - self.displayed_w) // 2
        self.canvas_offset_y = (ch - self.displayed_h) // 2

        rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        resized_pil = Image.fromarray(rgb).resize((self.displayed_w, self.displayed_h), Image.Resampling.BILINEAR)
        self.tk_image = ImageTk.PhotoImage(resized_pil)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas_offset_x,
            self.canvas_offset_y,
            anchor=tk.NW,
            image=self.tk_image
        )

        # Draw target box outline on canvas so user ALWAYS sees where the box is!
        for idx, (bx, by, bw, bh) in enumerate(self.detected_boxes, start=1):
            c_x1, c_y1 = self._img_to_canvas_coords(bx, by)
            c_x2, c_y2 = self._img_to_canvas_coords(bx + bw, by + bh)

            if self.current_view == "boxes":
                self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, outline="#00E676", width=2)
                self.canvas.create_text(
                    c_x1, max(12, c_y1 - 10),
                    text="Target: Watermark (Green Box)",
                    fill="#00E676",
                    anchor=tk.W,
                    font=("Segoe UI", 9, "bold")
                )
            else:
                # Cleaned view: draw subtle cyan dashed guide so user clearly sees the cleaned zone
                self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, outline="#00E5FF", width=1, dash=(4, 3))
                self.canvas.create_text(
                    c_x1, max(12, c_y1 - 10),
                    text="Cleaned Zone (Zero Blur)",
                    fill="#00E5FF",
                    anchor=tk.W,
                    font=("Segoe UI", 9, "bold")
                )

    # ----------------------------------------------------
    # FULL FILE PROCESSING & SAVING
    # ----------------------------------------------------
    def save_cleaned_file(self):
        if self.cv_image is None or not self.current_filepath:
            messagebox.showinfo("Info", "Please open a video or image file first.")
            return

        if self.is_processing:
            messagebox.showwarning("Busy", "Video rendering is already in progress. Please wait.")
            return

        if not self.detected_boxes:
            self.apply_preset("bottom_right")

        if not self.is_video:
            out_file = filedialog.asksaveasfilename(
                title="Save Cleaned Image",
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
            )
            if not out_file:
                return

            self.lbl_status.config(text="Saving clean image...")
            cleaned = clean_all_watermarks(
                self.cv_image,
                self.detected_boxes,
                mode=self.var_mode.get(),
                sensitivity=self.var_sensitivity.get()
            )
            cv2.imwrite(out_file, cleaned)
            self.lbl_status.config(text=f"Saved clean image: {os.path.basename(out_file)}")
            messagebox.showinfo("Success", f"Watermark removed and proper color restored!\n\nSaved to:\n{out_file}")

        else:
            out_file = filedialog.asksaveasfilename(
                title="Save Cleaned Video",
                defaultextension=".mp4",
                filetypes=[("MP4 Video", "*.mp4"), ("MKV Video", "*.mkv"), ("All Files", "*.*")]
            )
            if not out_file:
                return

            self.is_processing = True
            self.progress_bar["value"] = 0

            worker = threading.Thread(
                target=self._video_worker,
                args=(self.current_filepath, out_file, self.detected_boxes, self.cached_video_masks),
                daemon=True
            )
            worker.start()

    def _video_worker(self, in_file, out_file, boxes, precomputed_masks=None):
        def on_prog(curr, total):
            pct = (curr / total) * 100
            self.after(0, self._update_progress, curr, total, pct)

        try:
            success = process_video_clean(
                input_video_path=in_file,
                output_video_path=out_file,
                boxes=boxes,
                mode=self.var_mode.get(),
                sensitivity=self.var_sensitivity.get(),
                precomputed_masks=precomputed_masks,
                progress_callback=on_prog
            )
            self.after(0, self._on_video_complete, success, out_file)
        except Exception as e:
            self.after(0, self._on_video_error, str(e))

    def _update_progress(self, curr, total, pct):
        self.progress_bar["value"] = pct
        self.lbl_status.config(text=f"Rendering: Frame {curr}/{total} ({pct:.1f}%) - Proper Color")

    def _on_video_complete(self, success, out_file):
        self.is_processing = False
        self.progress_bar["value"] = 100
        if success and os.path.exists(out_file):
            self.lbl_status.config(text=f"Saved clean video: {os.path.basename(out_file)}")
            messagebox.showinfo("Success", f"Watermark removed and proper background color restored!\nOriginal audio is 100% intact.\n\nSaved to:\n{out_file}")
        else:
            self.lbl_status.config(text="Video processing failed.")
            messagebox.showerror("Error", "Video processing failed.")

    def _on_video_error(self, err_msg):
        self.is_processing = False
        self.lbl_status.config(text="Error occurred.")
        messagebox.showerror("Error", f"Error during video processing:\n{err_msg}")


def main():
    app = GeminiWatermarkRemover()
    app.mainloop()


if __name__ == "__main__":
    main()
