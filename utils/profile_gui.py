from __future__ import annotations

import argparse
import ast
import json
import sys
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

THIS_DIR = Path(__file__).resolve().parent
SESSION_FILE = THIS_DIR / ".profile_gui_session.json"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from profile_ops import (  # noqa: E402
    ProfileData,
    fixed_sep_exp_connection,
    fixed_sep_tanh_connection,
    profile_derivatives,
    read_prf,
    scale_psi_axis,
    smooth_pedestal_top,
    write_prf,
)


class ToolTip:
    def __init__(self, widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None

        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _show(self) -> None:
        if self._tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self._tip_window = tk.Toplevel(self.widget)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip_window,
            text=self.text,
            justify="left",
            wraplength=320,
            padx=8,
            pady=5,
            relief="solid",
            borderwidth=1,
            background="#ffffe8",
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None

    def _cancel(self) -> None:
        if self._after_id:
            self.widget.after_cancel(self._after_id)
            self._after_id = None


class ProfileEditorApp(tk.Tk):
    def __init__(
        self,
        initial_profile: Path | None = None,
        overlays: list[Path] | None = None,
        experiments: list[Path] | None = None,
        restore_session: bool = False,
    ):
        super().__init__()
        self.title("XGC Profile Editor")
        self.geometry("1400x900")
        self.minsize(1100, 720)

        self.profile_path: Path | None = None
        self.original: ProfileData | None = None
        self.current: ProfileData | None = None
        self.history: list[ProfileData] = []
        self.markers: list[float] = []
        self.patch_markers: list[float] = []
        self.overlays: list[ProfileData] = []
        self.experiments: list[ProfileData] = []
        self.experiment_scales: list[str] = []
        self._experiment_scale_editor = None

        self._build_vars()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.close)

        if restore_session:
            self.restore_session()
        elif initial_profile:
            self.load_profile(initial_profile)

        if not restore_session:
            for overlay in overlays or []:
                self.add_overlay(overlay)
            for experiment in experiments or []:
                self.add_experiment(experiment)

    def _build_vars(self) -> None:
        self.status_var = tk.StringVar(value="Open a .prf profile to begin.")
        self.current_points_var = tk.StringVar(value="Current grid points: -")
        self.show_original_var = tk.BooleanVar(value=True)
        self.show_previous_var = tk.BooleanVar(value=True)
        self.xmin_var = tk.StringVar(value="")
        self.xmax_var = tk.StringVar(value="")

        self.smooth_start_var = tk.StringVar(value="0.85")
        self.smooth_end_var = tk.StringVar(value="0.95")
        self.smooth_half_window_var = tk.StringVar(value="4")
        self.smooth_poly_deg_var = tk.StringVar(value="3")
        self.smooth_strength_var = tk.StringVar(value="0.4")
        self.smooth_patch_width_var = tk.StringVar(value="0.005")
        self.smooth_patch_passes_var = tk.StringVar(value="4")
        self.smooth_patch_alpha_var = tk.StringVar(value="0.25")

        self.shift_delta_var = tk.StringVar(value="-0.01")
        self.shift_ref_var = tk.StringVar(value="1.0")

        self.fixed_mode_var = tk.StringVar(value="tanh")
        self.fixed_start_var = tk.StringVar(value="0.95")
        self.fixed_sep_var = tk.StringVar(value="1.0")
        self.fixed_target_var = tk.StringVar(value="4.0e18")
        self.fixed_floor_var = tk.StringVar(value="2.5e18")
        self.fixed_width_var = tk.StringVar(value="0.05")
        self.fixed_k_var = tk.StringVar(value="-5.0")
        self.fixed_psimax_var = tk.StringVar(value="1.3")
        self.fixed_use_target_var = tk.BooleanVar(value=True)
        self.fixed_sep_var.trace_add("write", lambda *_: self._suggest_fixed_sep_values())
        self.fixed_psimax_var.trace_add("write", lambda *_: self._suggest_fixed_sep_values())

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        controls_holder = ttk.Frame(self)
        controls_holder.grid(row=0, column=0, sticky="ns")
        controls_holder.rowconfigure(0, weight=1)
        controls_holder.columnconfigure(0, weight=1)

        controls_canvas = tk.Canvas(controls_holder, width=330, highlightthickness=0)
        controls_scroll = ttk.Scrollbar(controls_holder, orient="vertical", command=controls_canvas.yview)
        controls_canvas.configure(yscrollcommand=controls_scroll.set)
        controls_canvas.grid(row=0, column=0, sticky="ns")
        controls_scroll.grid(row=0, column=1, sticky="ns")

        controls = ttk.Frame(controls_canvas, padding=10)
        controls.columnconfigure(0, weight=1)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")

        def update_scroll_region(_event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))

        def update_controls_width(event):
            controls_canvas.itemconfigure(controls_window, width=event.width)

        def scroll_controls(event):
            controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        controls.bind("<Configure>", update_scroll_region)
        controls_canvas.bind("<Configure>", update_controls_width)
        controls_canvas.bind("<MouseWheel>", scroll_controls)

        plot_frame = ttk.Frame(self, padding=(0, 10, 10, 0))
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self._build_file_controls(controls)
        control_tabs = ttk.Notebook(controls)
        control_tabs.grid(row=1, column=0, sticky="nsew")
        controls.rowconfigure(1, weight=1)

        modify_page = ttk.Frame(control_tabs, padding=(0, 8, 0, 0))
        overlay_page = ttk.Frame(control_tabs, padding=(0, 8, 0, 0))
        for page in (modify_page, overlay_page):
            page.columnconfigure(0, weight=1)

        control_tabs.add(modify_page, text="Modify")
        control_tabs.add(overlay_page, text="Overlays")

        self._build_operation_controls(modify_page, row=0)
        self._build_history_controls(modify_page, row=1)
        self._build_plot_controls(modify_page, row=2)
        self._build_overlay_controls(overlay_page, row=0)
        self._build_experiment_controls(overlay_page, row=1)
        self._build_plot(plot_frame)

        status = ttk.Label(self, textvariable=self.status_var, anchor="w", padding=(10, 4))
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    def _build_file_controls(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Profile", padding=8)
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)

        ttk.Button(frame, text="Open...", command=self.open_profile_dialog).grid(row=0, column=0, sticky="ew")
        ttk.Button(frame, text="Save As...", command=self.save_profile_dialog).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Reset", command=self.reset_profile).grid(row=2, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(frame, text="Undo", command=self.undo_profile).grid(row=3, column=0, sticky="ew", pady=(4, 0))

        self.profile_label = ttk.Label(frame, text="No profile loaded", width=34, wraplength=250)
        self.profile_label.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(frame, textvariable=self.current_points_var).grid(row=5, column=0, sticky="w", pady=(4, 0))

    def _build_operation_controls(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Modify", padding=8)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(row, weight=0)

        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky="ew")

        smooth_tab = ttk.Frame(notebook, padding=8)
        shift_tab = ttk.Frame(notebook, padding=8)
        fixed_tab = ttk.Frame(notebook, padding=8)
        notebook.add(shift_tab, text="Shift")
        notebook.add(fixed_tab, text="Fixed sep")
        notebook.add(smooth_tab, text="Smooth top")

        smooth_start_widgets = self._add_field(smooth_tab, "psi start", self.smooth_start_var, 0)
        smooth_end_widgets = self._add_field(smooth_tab, "psi end", self.smooth_end_var, 1)
        half_window_widgets = self._add_field(smooth_tab, "half window", self.smooth_half_window_var, 2)
        poly_degree_widgets = self._add_field(smooth_tab, "poly degree", self.smooth_poly_deg_var, 3)
        smooth_strength_widgets = self._add_field(smooth_tab, "smooth strength", self.smooth_strength_var, 4)
        patch_width_widgets = self._add_field(smooth_tab, "patch width", self.smooth_patch_width_var, 5)
        patch_passes_widgets = self._add_field(smooth_tab, "patch passes", self.smooth_patch_passes_var, 6)
        patch_alpha_widgets = self._add_field(smooth_tab, "patch alpha", self.smooth_patch_alpha_var, 7)
        self._add_tooltip(
            smooth_start_widgets,
            "Start of the interval replaced by the C1 cubic Hermite segment. Points below this psi are left unchanged.",
        )
        self._add_tooltip(
            smooth_end_widgets,
            "End of the interval replaced by the C1 cubic Hermite segment. Points above this psi are left unchanged.",
        )
        self._add_tooltip(
            half_window_widgets,
            "Number of grid points taken on each side of psi start/end when fitting local polynomials "
            "to estimate endpoint values and slopes.",
        )
        self._add_tooltip(
            poly_degree_widgets,
            "Polynomial degree for the local endpoint fits used only to estimate value and slope. "
            "The actual replacement segment is cubic Hermite.",
        )
        self._add_tooltip(
            smooth_strength_widgets,
            "Blends endpoint slopes toward the secant slope across the interval. "
            "0 preserves fitted endpoint slopes; 1 uses the secant slope at both ends.",
        )
        self._add_tooltip(
            patch_width_widgets,
            "Width in psi_N of the boundary region smoothed after the C1 operation. "
            "Diffusion passes are applied separately on [psi start - width, psi start + width] "
            "and [psi end - width, psi end + width], so each pass smooths across the original/C1 join.",
        )
        self._add_tooltip(
            patch_passes_widgets,
            "Number of explicit diffusion passes applied after the C1 operation inside each boundary patch. "
            "0 disables the post-smoothing pass.",
        )
        self._add_tooltip(
            patch_alpha_widgets,
            "Diffusion coefficient for each pass: new[i] = alpha*old[i-1] + (1-2*alpha)*old[i] + alpha*old[i+1]. "
            "Use 0 to 0.5; values around 0.2-0.3 are usually gentle and stable.",
        )
        ttk.Button(smooth_tab, text="Apply Smooth", command=self.apply_smooth).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        delta_psi_widgets = self._add_field(shift_tab, "delta psi", self.shift_delta_var, 0)
        psi_ref_widgets = self._add_field(shift_tab, "psi ref", self.shift_ref_var, 1)
        self._add_tooltip(
            delta_psi_widgets,
            "Scales the psi axis by (psi_ref - delta psi) / psi_ref. "
            "Profile values are copied unchanged; only psi coordinates move.",
        )
        self._add_tooltip(
            psi_ref_widgets,
            "Reference psi used in the shift scale factor. For psi_ref=1, negative delta psi expands "
            "the psi axis outward and positive delta psi compresses it inward.",
        )
        ttk.Button(shift_tab, text="Apply Shift", command=self.apply_shift).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))

        mode_label = ttk.Label(fixed_tab, text="mode")
        mode_label.grid(row=0, column=0, sticky="w", pady=2)
        mode = ttk.Combobox(fixed_tab, textvariable=self.fixed_mode_var, values=("tanh", "exp"), state="readonly", width=14)
        mode.grid(row=0, column=1, sticky="ew", pady=2)
        self._add_tooltip(
            (mode_label, mode),
            "Fixed-separatrix connection type. Tanh solves a tanh segment to hit target sep; "
            "exp uses an exponential segment and either solves k from target sep or uses exp k shape.",
        )
        fixed_start_widgets = self._add_field(fixed_tab, "psi start", self.fixed_start_var, 1)
        fixed_sep_widgets = self._add_field(fixed_tab, "psi sep", self.fixed_sep_var, 2)
        target_sep_widgets = self._add_field(fixed_tab, "target sep", self.fixed_target_var, 3)
        floor_widgets = self._add_field(fixed_tab, "floor", self.fixed_floor_var, 4)
        tanh_width_widgets = self._add_field(fixed_tab, "tanh width", self.fixed_width_var, 5)
        exp_k_widgets = self._add_field(fixed_tab, "exp k shape", self.fixed_k_var, 6)
        psi_max_widgets = self._add_field(fixed_tab, "psi max", self.fixed_psimax_var, 7)
        self.fixed_target_sep_widgets = target_sep_widgets
        self.fixed_tanh_width_widgets = tanh_width_widgets
        self.fixed_exp_k_widgets = exp_k_widgets
        self._add_tooltip(
            fixed_start_widgets,
            "Boundary between unchanged core and modified edge/SOL. Values for psi < psi start are preserved. "
            "The value and slope at psi start are estimated from a local linear fit.",
        )
        self._add_tooltip(
            fixed_sep_widgets,
            "Separatrix location for the connection. The modified edge segment is used up to this psi; "
            "beyond it the code applies an exponential SOL decay toward the floor value.",
        )
        self._add_tooltip(
            target_sep_widgets,
            "Target profile value at psi sep. Tanh mode always uses this. Exp mode uses it only when "
            "'Use target sep for exp' is checked; otherwise the sep value comes from exp k shape.",
        )
        self._add_tooltip(
            floor_widgets,
            "Asymptotic SOL floor value. After psi sep, the generated SOL branch decays exponentially "
            "from target sep toward this value.",
        )
        self._add_tooltip(
            tanh_width_widgets,
            "Used only in tanh mode. Sets the tanh transition width in psi_N. "
            "Larger values make the pedestal-to-separatrix connection broader and gentler; "
            "smaller values make it sharper and more localized.",
        )
        self._add_tooltip(
            exp_k_widgets,
            "Used in exp mode when 'Use target sep for exp' is unchecked. "
            "Controls the exponential shape between psi start and psi sep: value = A exp(k (psi - psi_start)) + C, "
            "with A chosen to match the fitted slope at psi start. Positive k makes the slope magnitude grow toward "
            "psi sep; negative k makes it relax toward psi sep. Around -5 is a gentle starting point for a 0.05-wide interval.",
        )
        self._add_tooltip(
            psi_max_widgets,
            "Maximum psi to generate for the modified profile. If the original profile ends before this, "
            "the grid is extended using the average spacing of the last few original points.",
        )
        use_target_check = ttk.Checkbutton(fixed_tab, text="Use target sep for exp", variable=self.fixed_use_target_var)
        use_target_check.grid(
            row=8, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        self.fixed_use_target_check = use_target_check
        self._add_tooltip(
            use_target_check,
            "Exp mode only. When checked, fsolve chooses k so the exponential segment hits target sep. "
            "When unchecked, exp k shape is used directly and the separatrix value is whatever that curve gives.",
        )
        self.fixed_mode_var.trace_add("write", lambda *_: self._update_fixed_sep_controls())
        self.fixed_use_target_var.trace_add("write", lambda *_: self._update_fixed_sep_controls())
        self._update_fixed_sep_controls()
        ttk.Button(fixed_tab, text="Apply Fixed Sep", command=self.apply_fixed_sep).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        for tab in (smooth_tab, shift_tab, fixed_tab):
            tab.columnconfigure(1, weight=1)

    def _build_history_controls(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Versions", padding=8)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(row, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.history_list = tk.Listbox(frame, height=5, exportselection=False)
        self.history_list.grid(row=0, column=0, sticky="nsew")
        self.history_list.bind("<<ListboxSelect>>", self.preview_history)
        self._add_tooltip(
            self.history_list,
            "Click a version to preview it as a dashed red curve. Histories are in-memory only and are not restored after closing.",
        )
        restore_button = ttk.Button(frame, text="Restore Selected", command=self.restore_history_selection)
        restore_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._add_tooltip(
            restore_button,
            "Replace the current modified profile with the selected in-memory version and discard later history entries.",
        )

    def _build_overlay_controls(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Overlays", padding=8)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(row, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        add_overlay_button = ttk.Button(frame, text="Add Overlay...", command=self.add_overlay_dialog)
        add_overlay_button.grid(row=0, column=0, sticky="ew")
        self.overlay_list = tk.Listbox(frame, height=4, exportselection=False)
        self.overlay_list.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        remove_overlay_button = ttk.Button(frame, text="Remove Selected", command=self.remove_overlay_selection)
        remove_overlay_button.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        overlay_tip = (
            "Add comparison profiles as line plots. Overlays are drawn on Value, dValue/dpsi, and d2Value/dpsi2, "
            "so their psi grid must be strictly increasing."
        )
        self._add_tooltip((add_overlay_button, self.overlay_list), overlay_tip)
        self._add_tooltip(remove_overlay_button, "Remove the selected overlay profile from the comparison plot.")

    def _build_experiment_controls(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Experimental", padding=8)
        frame.grid(row=row, column=0, sticky="nsew", pady=(0, 8))
        parent.rowconfigure(row, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        add_experiment_button = ttk.Button(frame, text="Add Experiment...", command=self.add_experiment_dialog)
        add_experiment_button.grid(row=0, column=0, sticky="ew")
        self.experiment_table = ttk.Treeview(
            frame,
            columns=("scale",),
            height=5,
            selectmode="extended",
        )
        self.experiment_table.heading("#0", text="File")
        self.experiment_table.heading("scale", text="Scale")
        self.experiment_table.column("#0", width=230, minwidth=120, stretch=True)
        self.experiment_table.column("scale", width=70, minwidth=55, stretch=False, anchor="center")
        self.experiment_table.grid(row=1, column=0, sticky="nsew", pady=(4, 0))
        self.experiment_table.bind("<Double-1>", self._edit_experiment_scale)
        remove_experiment_button = ttk.Button(frame, text="Remove Selected", command=self.remove_experiment_selection)
        remove_experiment_button.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        experiment_tip = (
            "Add experimental/reference points as filled square markers on the Value panel only. "
            "Repeated psi values are allowed because no derivatives are computed for these files. "
            "Double-click the Scale cell to enter an optional scalar or expression such as 1/3; "
            "a scaled copy is drawn as hollow squares."
        )
        self._add_tooltip((add_experiment_button, self.experiment_table), experiment_tip)
        self._add_tooltip(remove_experiment_button, "Remove the selected experimental point set from the Value panel.")

    def _build_plot_controls(self, parent: ttk.Frame, row: int) -> None:
        frame = ttk.LabelFrame(parent, text="Plot", padding=8)
        frame.grid(row=row, column=0, sticky="ew")
        frame.columnconfigure(1, weight=1)

        original_check = ttk.Checkbutton(frame, text="Original", variable=self.show_original_var, command=self.refresh_plot)
        original_check.grid(
            row=0, column=0, sticky="w"
        )
        previous_check = ttk.Checkbutton(frame, text="Previous", variable=self.show_previous_var, command=self.refresh_plot)
        previous_check.grid(
            row=0, column=1, sticky="w"
        )
        self._add_tooltip(original_check, "Show or hide the profile exactly as it was loaded from disk.")
        self._add_tooltip(previous_check, "Show or hide the immediately previous in-memory version before the current operation.")
        xmin_widgets = self._add_field(frame, "xmin", self.xmin_var, 1)
        xmax_widgets = self._add_field(frame, "xmax", self.xmax_var, 2)
        self._add_tooltip(
            xmin_widgets,
            "Optional shared left x-limit for all three panels. Leave blank to keep the current/autoscaled limit.",
        )
        self._add_tooltip(
            xmax_widgets,
            "Optional shared right x-limit for all three panels. Leave blank to keep the current/autoscaled limit.",
        )
        ttk.Button(frame, text="Refresh Plot", command=self.refresh_plot).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))

    def _build_plot(self, parent: ttk.Frame) -> None:
        self.figure = Figure(figsize=(10, 7), dpi=100)
        grid = self.figure.add_gridspec(
            2,
            2,
            width_ratios=[1.15, 1.0],
            height_ratios=[1.0, 1.0],
            left=0.06,
            right=0.98,
            bottom=0.08,
            top=0.80,
            wspace=0.24,
            hspace=0.30,
        )
        ax_value = self.figure.add_subplot(grid[:, 0])
        ax_d1 = self.figure.add_subplot(grid[0, 1], sharex=ax_value)
        ax_d2 = self.figure.add_subplot(grid[1, 1], sharex=ax_value)
        self.legend = None
        self.axes = [ax_value, ax_d1, ax_d2]
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        toolbar.grid(row=1, column=0, sticky="ew")
        self.refresh_plot(preserve_view=False)

    def _add_field(self, parent: ttk.Frame, label: str, var: tk.StringVar, row: int):
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=2)
        entry_widget = ttk.Entry(parent, textvariable=var, width=14)
        entry_widget.grid(row=row, column=1, sticky="ew", pady=2)
        return label_widget, entry_widget

    def _add_tooltip(self, widgets, text: str) -> None:
        if not isinstance(widgets, (tuple, list)):
            widgets = (widgets,)
        for widget in widgets:
            ToolTip(widget, text)

    def _set_field_enabled(self, field_widgets, enabled: bool) -> None:
        _label, entry = field_widgets
        entry.configure(state="normal" if enabled else "disabled")

    def _set_widget_enabled(self, widget, enabled: bool) -> None:
        if hasattr(widget, "state"):
            widget.state(["!disabled"] if enabled else ["disabled"])
        else:
            widget.configure(state="normal" if enabled else "disabled")

    def _update_fixed_sep_controls(self) -> None:
        mode = self.fixed_mode_var.get()
        use_target_for_exp = self.fixed_use_target_var.get()

        is_tanh = mode == "tanh"
        is_exp = mode == "exp"

        self._set_field_enabled(self.fixed_tanh_width_widgets, is_tanh)
        self._set_widget_enabled(self.fixed_use_target_check, is_exp)
        self._set_field_enabled(self.fixed_target_sep_widgets, is_tanh or (is_exp and use_target_for_exp))
        self._set_field_enabled(self.fixed_exp_k_widgets, is_exp and not use_target_for_exp)

    def open_profile_dialog(self) -> None:
        filename = filedialog.askopenfilename(
            title="Open XGC profile",
            filetypes=(("Profile files", "*.prf"), ("All files", "*.*")),
        )
        if filename:
            self.load_profile(Path(filename))

    def load_profile(self, path: Path) -> None:
        try:
            profile = read_prf(path)
        except Exception as exc:
            messagebox.showerror("Could not open profile", str(exc))
            return

        self.profile_path = path
        self.original = profile
        self.current = ProfileData(profile.psi.copy(), profile.value.copy(), label="Current", path=path)
        self.history = [ProfileData(profile.psi.copy(), profile.value.copy(), label="Original", path=path)]
        self.markers = []
        self.patch_markers = []
        self.profile_label.configure(text=str(path))
        self._update_profile_count()
        self._suggest_fixed_sep_values()
        self._refresh_history_list()
        self.status_var.set(f"Loaded {path.name} with {len(profile.psi)} points.")
        self.refresh_plot(preserve_view=False)

    def save_profile_dialog(self) -> None:
        if not self.current:
            messagebox.showinfo("No profile", "Open a profile before saving.")
            return

        initial = "modified.prf"
        if self.profile_path:
            initial = f"{self.profile_path.stem}_edited{self.profile_path.suffix}"
        filename = filedialog.asksaveasfilename(
            title="Save modified profile",
            initialfile=initial,
            defaultextension=".prf",
            filetypes=(("Profile files", "*.prf"), ("All files", "*.*")),
        )
        if not filename:
            return

        try:
            write_prf(filename, self.current.psi, self.current.value)
        except Exception as exc:
            messagebox.showerror("Could not save profile", str(exc))
            return
        self.status_var.set(f"Saved {filename}.")

    def reset_profile(self) -> None:
        if not self.original:
            return
        self.current = ProfileData(self.original.psi.copy(), self.original.value.copy(), label="Current", path=self.original.path)
        self.history = [ProfileData(self.original.psi.copy(), self.original.value.copy(), label="Original", path=self.original.path)]
        self.markers = []
        self.patch_markers = []
        self._update_profile_count()
        self._suggest_fixed_sep_values()
        self._refresh_history_list()
        self.status_var.set("Reset to original profile.")
        self.refresh_plot(preserve_view=False)

    def undo_profile(self) -> None:
        if len(self.history) <= 1:
            return
        self.history.pop()
        last = self.history[-1]
        self.current = ProfileData(last.psi.copy(), last.value.copy(), label="Current", path=last.path)
        self.markers = []
        self.patch_markers = []
        self._update_profile_count()
        self._refresh_history_list()
        self.status_var.set(f"Restored {last.label}.")
        self._suggest_fixed_sep_values()
        self.refresh_plot()

    def apply_smooth(self) -> None:
        if not self._require_profile():
            return
        try:
            result = smooth_pedestal_top(
                self.current.psi,
                self.current.value,
                psi_start=self._float(self.smooth_start_var, "psi start"),
                psi_end=self._float(self.smooth_end_var, "psi end"),
                half_window=self._int(self.smooth_half_window_var, "half window"),
                poly_deg=self._int(self.smooth_poly_deg_var, "poly degree"),
                smooth_strength=self._float(self.smooth_strength_var, "smooth strength"),
                patch_width=self._float(self.smooth_patch_width_var, "patch width"),
                patch_passes=self._int(self.smooth_patch_passes_var, "patch passes"),
                patch_alpha=self._float(self.smooth_patch_alpha_var, "patch alpha"),
            )
        except Exception as exc:
            self._show_apply_error(exc)
            return
        self._accept_result(result.psi, result.value, "Smooth top", result.markers, result.summary, result.patch_markers)

    def apply_shift(self) -> None:
        if not self._require_profile():
            return
        try:
            result = scale_psi_axis(
                self.current.psi,
                self.current.value,
                delta_psi=self._float(self.shift_delta_var, "delta psi"),
                psi_ref=self._float(self.shift_ref_var, "psi ref"),
            )
        except Exception as exc:
            self._show_apply_error(exc)
            return
        self._accept_result(result.psi, result.value, "Shift", result.markers, result.summary)

    def apply_fixed_sep(self) -> None:
        if not self._require_profile():
            return
        try:
            kwargs = {
                "psi_start": self._float(self.fixed_start_var, "psi start"),
                "psi_sep": self._float(self.fixed_sep_var, "psi sep"),
                "floor_val": self._float(self.fixed_floor_var, "floor"),
                "psi_max": self._float(self.fixed_psimax_var, "psi max"),
            }
            if self.fixed_mode_var.get() == "tanh":
                result = fixed_sep_tanh_connection(
                    self.current.psi,
                    self.current.value,
                    target_val_sep=self._float(self.fixed_target_var, "target sep"),
                    w_tanh=self._float(self.fixed_width_var, "tanh width"),
                    **kwargs,
                )
                label = "Fixed sep tanh"
            else:
                target = self._float(self.fixed_target_var, "target sep") if self.fixed_use_target_var.get() else None
                result = fixed_sep_exp_connection(
                    self.current.psi,
                    self.current.value,
                    target_val_sep=target,
                    k_shape=self._float(self.fixed_k_var, "exp k shape"),
                    **kwargs,
                )
                label = "Fixed sep exp"
        except Exception as exc:
            self._show_apply_error(exc)
            return
        self._accept_result(result.psi, result.value, label, result.markers, result.summary)

    def _accept_result(
        self,
        psi,
        value,
        label: str,
        markers: list[float],
        summary: str,
        patch_markers: list[float] | None = None,
    ) -> None:
        self.current = ProfileData(psi.copy(), value.copy(), label="Current", path=self.profile_path)
        version_label = f"{len(self.history)}: {label}"
        self.history.append(ProfileData(psi.copy(), value.copy(), label=version_label, path=self.profile_path))
        self.markers = markers
        self.patch_markers = patch_markers or []
        self._update_profile_count()
        self._suggest_fixed_sep_values()
        self._refresh_history_list()
        self.status_var.set(summary)
        self.refresh_plot()

    def preview_history(self, _event=None) -> None:
        self.refresh_plot()

    def restore_history_selection(self) -> None:
        selection = self.history_list.curselection()
        if not selection:
            return
        idx = selection[0]
        selected = self.history[idx]
        self.current = ProfileData(selected.psi.copy(), selected.value.copy(), label="Current", path=selected.path)
        self.history = self.history[: idx + 1]
        self.markers = []
        self.patch_markers = []
        self._update_profile_count()
        self._suggest_fixed_sep_values()
        self._refresh_history_list()
        self.status_var.set(f"Restored {selected.label}.")
        self.refresh_plot()

    def _refresh_history_list(self) -> None:
        self.history_list.delete(0, tk.END)
        for item in self.history:
            self.history_list.insert(tk.END, item.label)
        if self.history:
            self.history_list.selection_clear(0, tk.END)
            self.history_list.selection_set(len(self.history) - 1)
            self.history_list.see(len(self.history) - 1)

    def add_overlay_dialog(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Add overlay profiles",
            filetypes=(("Profile files", "*.prf"), ("All files", "*.*")),
        )
        for filename in filenames:
            self.add_overlay(Path(filename))

    def add_overlay(self, path: Path) -> None:
        try:
            profile = read_prf(path)
        except Exception as exc:
            messagebox.showerror("Could not open overlay", str(exc))
            return
        self.overlays.append(profile)
        self.overlay_list.insert(tk.END, profile.label)
        self.status_var.set(f"Added overlay {profile.label}.")
        self.refresh_plot()

    def remove_overlay_selection(self) -> None:
        selection = list(self.overlay_list.curselection())
        if not selection:
            return
        for idx in reversed(selection):
            self.overlay_list.delete(idx)
            self.overlays.pop(idx)
        self.refresh_plot()

    def add_experiment_dialog(self) -> None:
        filenames = filedialog.askopenfilenames(
            title="Add experimental profiles",
            filetypes=(("Profile files", "*.prf"), ("All files", "*.*")),
        )
        for filename in filenames:
            self.add_experiment(Path(filename))

    def add_experiment(self, path: Path, scale: str = "") -> None:
        try:
            profile = read_prf(path, require_strict_psi=False)
        except Exception as exc:
            messagebox.showerror("Could not open experimental profile", str(exc))
            return
        self.experiments.append(profile)
        self.experiment_scales.append(scale.strip())
        self._refresh_experiment_table()
        self.status_var.set(f"Added experimental profile {profile.label}.")
        self.refresh_plot()

    def restore_session(self) -> None:
        if not SESSION_FILE.exists():
            return

        try:
            data = json.loads(SESSION_FILE.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self.status_var.set(f"Could not restore previous session: {exc}")
            return

        profile_path = self._path_from_session(data.get("profile"))
        if profile_path:
            self.load_profile(profile_path)

        for item in data.get("overlays", []):
            path = self._path_from_session(item)
            if path:
                self.add_overlay(path)

        for item in data.get("experiments", []):
            scale = ""
            path_item = item
            if isinstance(item, dict):
                path_item = item.get("path")
                scale = str(item.get("scale", "")).strip()
            path = self._path_from_session(path_item)
            if path:
                self.add_experiment(path, scale=scale)

        closed_at = data.get("closed_at", "")
        if self.current:
            self.status_var.set(f"Restored previous session. Last closed: {closed_at}")

    def save_session(self) -> None:
        data = {
            "closed_at": datetime.now().isoformat(timespec="seconds"),
            "profile": str(self.profile_path) if self.profile_path else None,
            "overlays": [str(profile.path) for profile in self.overlays if profile.path],
            "experiments": [
                {"path": str(profile.path), "scale": self.experiment_scales[i] if i < len(self.experiment_scales) else ""}
                for i, profile in enumerate(self.experiments)
                if profile.path
            ],
        }
        try:
            SESSION_FILE.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            self.status_var.set(f"Could not save previous session: {exc}")

    def close(self) -> None:
        self.save_session()
        self.destroy()

    def _path_from_session(self, value) -> Path | None:
        if not value:
            return None
        path = Path(value)
        if not path.exists():
            return None
        return path

    def remove_experiment_selection(self) -> None:
        self._cancel_experiment_scale_edit()
        selection = list(self.experiment_table.selection())
        if not selection:
            return
        for idx in sorted((int(item) for item in selection), reverse=True):
            self.experiments.pop(idx)
            self.experiment_scales.pop(idx)
        self._refresh_experiment_table()
        self.refresh_plot()

    def _refresh_experiment_table(self) -> None:
        if not hasattr(self, "experiment_table"):
            return
        self._cancel_experiment_scale_edit()
        self.experiment_table.delete(*self.experiment_table.get_children())
        for i, experiment in enumerate(self.experiments):
            scale = self.experiment_scales[i] if i < len(self.experiment_scales) else ""
            self.experiment_table.insert("", "end", iid=str(i), text=experiment.label, values=(scale,))

    def _edit_experiment_scale(self, event) -> None:
        if self.experiment_table.identify_region(event.x, event.y) != "cell":
            return
        column = self.experiment_table.identify_column(event.x)
        if column != "#1":
            return
        iid = self.experiment_table.identify_row(event.y)
        if not iid:
            return
        bbox = self.experiment_table.bbox(iid, column)
        if not bbox:
            return

        self._cancel_experiment_scale_edit()
        x, y, width, height = bbox
        entry = ttk.Entry(self.experiment_table)
        entry.insert(0, self.experiment_scales[int(iid)])
        entry.select_range(0, tk.END)
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()
        self._experiment_scale_editor = entry

        def commit(_event=None):
            if self._experiment_scale_editor is not entry:
                return "break"
            text = entry.get().strip()
            if text:
                try:
                    self._parse_scale_expression(text)
                except ValueError:
                    self.status_var.set("Experimental scale must be numeric, e.g. 0.5 or 1/3, or blank to hide scaled points.")
                    entry.focus_set()
                    entry.select_range(0, tk.END)
                    return "break"
            self.experiment_scales[int(iid)] = text
            self._cancel_experiment_scale_edit()
            self._refresh_experiment_table()
            self.refresh_plot()
            return "break"

        def cancel(_event=None):
            if self._experiment_scale_editor is not entry:
                return "break"
            self._cancel_experiment_scale_edit()
            return "break"

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", cancel)

    def _cancel_experiment_scale_edit(self) -> None:
        if self._experiment_scale_editor is not None:
            editor = self._experiment_scale_editor
            self._experiment_scale_editor = None
            editor.destroy()

    def refresh_plot(self, preserve_view: bool = True) -> None:
        view_limits = self._capture_view_limits() if preserve_view else None

        for ax in self.axes:
            ax.clear()
        if self.legend:
            self.legend.remove()
            self.legend = None

        if not self.current:
            self.axes[0].set_title("No profile loaded")
            self.axes[0].set_ylabel("")
            self.axes[0].set_xlabel("psi_N")
            self.axes[1].set_title("dValue/dpsi")
            self.axes[1].set_ylabel("")
            self.axes[2].set_title("d2Value/dpsi2")
            self.axes[2].set_ylabel("")
            self.axes[2].set_xlabel("psi_N")
            self.canvas.draw_idle()
            return

        selected_preview = self._selected_history_profile()
        previous = self.history[-2] if len(self.history) >= 2 else None

        if self.original and self.show_original_var.get():
            self._plot_profile(self.original, "Original", color="black", linestyle="--", alpha=0.55, zorder=4)
        if previous and self.show_previous_var.get():
            self._plot_profile(previous, "Previous", color="tab:gray", linestyle=":", alpha=0.85, zorder=4)
        if selected_preview and selected_preview is not self.history[-1]:
            self._plot_profile(selected_preview, selected_preview.label, color="tab:red", linestyle="--", alpha=0.75, linewidth=2.0, zorder=6)

        self._plot_profile(
            self.current,
            f"Current (grid points: {len(self.current.psi)})",
            color="tab:red",
            linestyle="-",
            linewidth=2.0,
            zorder=7,
        )

        overlay_colors = ["b", "g", "c", "y", "tab:orange", "tab:purple", "tab:brown", "tab:pink"]
        for i, overlay in enumerate(self.overlays):
            color = overlay_colors[i % len(overlay_colors)]
            self._plot_profile(overlay, overlay.label, color=color, linestyle="-", linewidth=2.0, alpha=0.9, zorder=3)

        exp_colors = ["b", "r", "g", "m", "c", "y"]
        for i, experiment in enumerate(self.experiments):
            color = exp_colors[i % len(exp_colors)]
            scale_text = self.experiment_scales[i] if i < len(self.experiment_scales) else ""
            self.axes[0].scatter(
                experiment.psi,
                experiment.value,
                s=28,
                marker="s",
                facecolors=color,
                edgecolors=color,
                linewidths=0.35,
                alpha=0.55,
                label=f"Exp: {experiment.label}",
                zorder=8,
            )
            if scale_text:
                try:
                    scale = self._parse_scale_expression(scale_text)
                except ValueError:
                    scale = None
                if scale is not None:
                    self.axes[0].scatter(
                        experiment.psi,
                        experiment.value * scale,
                        s=38,
                        marker="s",
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.2,
                        alpha=0.55,
                        label=f"Exp x{scale:g}: {experiment.label}",
                        zorder=9,
                    )

        for marker in self.markers:
            for ax in self.axes:
                ax.axvline(marker, color="tab:blue", linestyle=":", linewidth=2.2, alpha=0.95)
        for marker in self.patch_markers:
            for ax in self.axes:
                ax.axvline(marker, color="tab:green", linestyle=":", linewidth=2.0, alpha=0.95)
        if self.current and len(self.patch_markers) == 4:
            g0, g1, g2, g3 = self.patch_markers
            patch_mask = ((self.current.psi >= g0) & (self.current.psi <= g1)) | ((self.current.psi >= g2) & (self.current.psi <= g3))
            if patch_mask.any():
                d1, d2 = profile_derivatives(self.current.psi, self.current.value)
                for ax, y_values, label in (
                    (self.axes[0], self.current.value, "Diffusion-zone points"),
                    (self.axes[1], d1, None),
                    (self.axes[2], d2, None),
                ):
                    ax.scatter(
                        self.current.psi[patch_mask],
                        y_values[patch_mask],
                        s=70,
                        marker="$*$",
                        facecolors="tab:green",
                        edgecolors="tab:green",
                        linewidths=0.8,
                        alpha=0.9,
                        label=label,
                        zorder=8.5,
                    )
        for ax in self.axes:
            ax.axvline(1.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.axhline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.7)
            ax.grid(True, which="major", alpha=0.35)
            ax.minorticks_on()
            ax.grid(True, which="minor", linestyle=":", alpha=0.2)

        self.axes[0].set_title("Value")
        self.axes[0].set_ylabel("")
        self.axes[0].set_xlabel("psi_N")
        self.axes[1].set_title("dValue/dpsi")
        self.axes[1].set_ylabel("")
        self.axes[2].set_title("d2Value/dpsi2")
        self.axes[2].set_ylabel("")
        self.axes[2].set_xlabel("psi_N")
        self._draw_legend()

        self._restore_view_limits(view_limits)
        self._apply_xlim()
        self.canvas.draw_idle()

    def _plot_profile(
        self,
        profile: ProfileData,
        label: str,
        *,
        color: str,
        linestyle: str,
        alpha: float = 1.0,
        linewidth: float = 1.5,
        zorder: int = 3,
    ) -> None:
        d1, d2 = profile_derivatives(profile.psi, profile.value)
        self.axes[0].plot(profile.psi, profile.value, label=label, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=zorder)
        self.axes[1].plot(profile.psi, d1, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=zorder)
        self.axes[2].plot(profile.psi, d2, color=color, linestyle=linestyle, alpha=alpha, linewidth=linewidth, zorder=zorder)

    def _draw_legend(self) -> None:
        handles, labels = self.axes[0].get_legend_handles_labels()
        if not handles:
            return

        unique_handles = []
        unique_labels = []
        seen = set()
        for handle, label in zip(handles, labels):
            if label in seen:
                continue
            seen.add(label)
            unique_handles.append(handle)
            unique_labels.append(label)

        self.legend = self.figure.legend(
            unique_handles,
            unique_labels,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.985),
            frameon=False,
            fontsize="small",
            borderaxespad=0.0,
            handlelength=2.4,
            labelspacing=0.6,
            columnspacing=1.8,
            ncol=2,
        )

    def _apply_xlim(self) -> None:
        xmin = self.xmin_var.get().strip()
        xmax = self.xmax_var.get().strip()
        if not xmin and not xmax:
            return
        try:
            left = float(xmin) if xmin else None
            right = float(xmax) if xmax else None
        except ValueError:
            self.status_var.set("Plot x-limits ignored: xmin/xmax must be numeric.")
            return
        for ax in self.axes:
            ax.set_xlim(left=left, right=right)

    def _capture_view_limits(self):
        if not any(ax.has_data() for ax in self.axes):
            return None
        return [(ax.get_xlim(), ax.get_ylim()) for ax in self.axes]

    def _restore_view_limits(self, view_limits) -> None:
        if not view_limits:
            return
        for ax, (xlim, ylim) in zip(self.axes, view_limits):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def _selected_history_profile(self) -> ProfileData | None:
        selection = self.history_list.curselection()
        if not selection:
            return None
        idx = selection[0]
        if 0 <= idx < len(self.history):
            return self.history[idx]
        return None

    def _require_profile(self) -> bool:
        if self.current is not None:
            return True
        messagebox.showinfo("No profile", "Open a profile before applying a modification.")
        return False

    def _update_profile_count(self) -> None:
        if self.current is None:
            self.current_points_var.set("Current grid points: -")
        else:
            self.current_points_var.set(f"Current grid points: {len(self.current.psi)}")

    def _suggest_fixed_sep_values(self) -> None:
        if self.current is None:
            return
        try:
            psi_sep = float(self.fixed_sep_var.get())
            psi_max = float(self.fixed_psimax_var.get())
        except ValueError:
            return
        target_sep = self._sample_current_profile(psi_sep)
        floor = self._sample_current_profile(psi_max)
        self.fixed_target_var.set(self._format_suggested_value(target_sep))
        self.fixed_floor_var.set(self._format_suggested_value(floor))

    def _sample_current_profile(self, psi_value: float) -> float:
        return float(np.interp(psi_value, self.current.psi, self.current.value))

    def _format_suggested_value(self, value: float) -> str:
        if value == 0.0:
            return "0.000"
        if abs(value) >= 1.0e5 or abs(value) < 1.0e-2:
            return f"{value:.3e}"
        return f"{value:.3f}"

    def _parse_scale_expression(self, text: str) -> float:
        allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        allowed_unary = (ast.UAdd, ast.USub)

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_binops):
                left = eval_node(node.left)
                right = eval_node(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                return left**right
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary):
                value = eval_node(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            raise ValueError("unsupported scale expression")

        try:
            value = eval_node(ast.parse(text, mode="eval"))
        except Exception as exc:
            raise ValueError("unsupported scale expression") from exc
        if not (-1.0e300 < value < 1.0e300):
            raise ValueError("scale expression is not finite")
        return value

    def _float(self, var: tk.StringVar, label: str) -> float:
        try:
            return float(var.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number") from exc

    def _int(self, var: tk.StringVar, label: str) -> int:
        try:
            return int(var.get())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer") from exc

    def _show_apply_error(self, exc: Exception) -> None:
        self.status_var.set(f"Modification failed: {exc}")
        messagebox.showerror("Modification failed", str(exc))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GUI editor for XGC .prf profile modifications.")
    parser.add_argument("profile", nargs="?", type=Path, help="Initial .prf profile to open")
    parser.add_argument("--overlay", "-o", action="append", type=Path, default=[], help="Profile overlay plotted as value/d1/d2 lines")
    parser.add_argument("--experiment", "-e", action="append", type=Path, default=[], help="Experimental points plotted on the value panel")
    parser.add_argument("--no-restore", action="store_true", help="Start empty instead of restoring the previous no-argument session")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restore_session = not args.no_restore and not args.profile and not args.overlay and not args.experiment
    app = ProfileEditorApp(
        initial_profile=args.profile,
        overlays=args.overlay,
        experiments=args.experiment,
        restore_session=restore_session,
    )
    app.mainloop()


if __name__ == "__main__":
    main()
