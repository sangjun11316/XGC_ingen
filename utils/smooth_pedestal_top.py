import numpy as np
import matplotlib.pyplot as plt
import sys, os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf
except ImportError as e:
    print(f"Warning: Could not import 'xgc_ingen'. Using local read_prf fallback. Details: {e}")

    def read_prf(filename):
        """
        Fallback reader for TOMMS-style .prf:
            line 1: number of points
            next N lines: psi  value
            final line: -1
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        with open(filename, "r") as f:
            n = int(f.readline().strip())
            psi = np.zeros(n, dtype=float)
            val = np.zeros(n, dtype=float)

            for i in range(n):
                parts = f.readline().split()
                psi[i] = float(parts[0])
                val[i] = float(parts[1])

        return psi, val


# --- Helper Functions ---
def write_prf(filename, psi, value):
    """Writes data to a .prf file in TOMMS format."""
    try:
        with open(filename, 'w') as f:
            f.write(f"{len(psi)}\n")
            for p, v in zip(psi, value):
                f.write(f"{p:.12E}  {v:.12E}\n")
            f.write("-1\n")
        print(f">> Written to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")


class ProfileModifier:
    def __init__(self, filename):
        self.filename = filename
        self.psi, self.val = read_prf(filename)

        self.psi_orig = self.psi.copy()
        self.val_orig = self.val.copy()

    def _get_local_value_and_slope(self, psi_ref, half_window=4, poly_deg=3):
        """
        Estimate value and 1st derivative at psi_ref from a local polynomial fit.
        """
        idx = np.argmin(np.abs(self.psi_orig - psi_ref))
        i0 = max(0, idx - half_window)
        i1 = min(len(self.psi_orig), idx + half_window + 1)

        p_local = self.psi_orig[i0:i1]
        v_local = self.val_orig[i0:i1]

        deg = min(poly_deg, len(p_local) - 1)
        if deg < 1:
            raise ValueError("Need at least 2 local points to estimate slope.")

        coef = np.polyfit(p_local, v_local, deg)
        poly = np.poly1d(coef)
        d1 = np.polyder(poly, 1)

        val_ref = poly(psi_ref)
        grad_ref = d1(psi_ref)

        return val_ref, grad_ref

    def _build_cubic_c1_segment(self, psi_start, psi_end, y0, dy0, y1, dy1, smooth_strength=0.0):
        """
        Build a cubic Hermite segment on [psi_start, psi_end] matching value and 1st derivative.

        smooth_strength:
            0.0 -> use original endpoint slopes
            1.0 -> move endpoint slopes fully to secant slope
        """
        dpsi = psi_end - psi_start
        if dpsi <= 0:
            raise ValueError("psi_end must be greater than psi_start.")

        secant = (y1 - y0) / dpsi

        # Blend endpoint slopes toward secant slope
        # stronger smoothing => flatter, less kinked pedestal top
        dy0_mod = (1.0 - smooth_strength) * dy0 + smooth_strength * secant
        dy1_mod = (1.0 - smooth_strength) * dy1 + smooth_strength * secant

        def func(psi):
            psi = np.asarray(psi)
            t = (psi - psi_start) / dpsi

            h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
            h10 = t**3 - 2.0 * t**2 + t
            h01 = -2.0 * t**3 + 3.0 * t**2
            h11 = t**3 - t**2

            return (
                h00 * y0
                + h10 * dpsi * dy0_mod
                + h01 * y1
                + h11 * dpsi * dy1_mod
            )

        return func, dy0_mod, dy1_mod, secant

    def smooth_pedestal_top(self, psi_start, psi_end, half_window=4, poly_deg=3, smooth_strength=0.0):
        """
        Smooth the interval psi_start <= psi <= psi_end with a C1-matched cubic Hermite segment.

        smooth_strength:
            0.0 -> minimal change, preserve endpoint slopes
            1.0 -> endpoint slopes become secant slope
        """
        if psi_start >= psi_end:
            raise ValueError("psi_start must be smaller than psi_end.")

        if psi_start < self.psi_orig.min() or psi_end > self.psi_orig.max():
            raise ValueError("Requested interval lies outside the original grid.")

        if not (0.0 <= smooth_strength <= 1.0):
            raise ValueError("smooth_strength must be between 0 and 1.")

        y0, dy0 = self._get_local_value_and_slope(psi_start, half_window=half_window, poly_deg=poly_deg)
        y1, dy1 = self._get_local_value_and_slope(psi_end,   half_window=half_window, poly_deg=poly_deg)

        func_mid, dy0_mod, dy1_mod, secant = self._build_cubic_c1_segment(
            psi_start, psi_end, y0, dy0, y1, dy1, smooth_strength=smooth_strength
        )

        print(">> Smoothing pedestal top with C1 cubic Hermite segment:")
        print(f"   Start: psi={psi_start:.5f}, val={y0:.6e}, grad(orig)={dy0:.6e}, grad(mod)={dy0_mod:.6e}")
        print(f"   End  : psi={psi_end:.5f}, val={y1:.6e}, grad(orig)={dy1:.6e}, grad(mod)={dy1_mod:.6e}")
        print(f"   Secant slope = {secant:.6e}")
        print(f"   smooth_strength = {smooth_strength:.3f}")

        psi_new = self.psi_orig.copy()
        val_new = self.val_orig.copy()

        mask_mid = (psi_new >= psi_start) & (psi_new <= psi_end)
        val_new[mask_mid] = func_mid(psi_new[mask_mid])

        self.psi = psi_new
        self.val = val_new
        self.trans_pts = [psi_start, psi_end]

    def plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # value
        axs[0].plot(self.psi_orig, self.val_orig, 'k--', label='Original', alpha=0.5)
        axs[0].plot(self.psi, self.val, 'r-', lw=2, label='Modified')
        if hasattr(self, 'trans_pts'):
            axs[0].axvline(self.trans_pts[0], c='g', ls=':', label='Start')
            axs[0].axvline(self.trans_pts[1], c='b', ls=':', label='End')
        axs[0].set_ylabel('Value')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        axs[0].set_title('Profile')

        # first derivative
        d_orig = np.gradient(self.val_orig, self.psi_orig)
        d_mod = np.gradient(self.val, self.psi)
        axs[1].plot(self.psi_orig, d_orig, 'k--', alpha=0.5, label='Original')
        axs[1].plot(self.psi, d_mod, 'r-', lw=2, label='Modified')
        if hasattr(self, 'trans_pts'):
            axs[1].axvline(self.trans_pts[0], c='g', ls=':')
            axs[1].axvline(self.trans_pts[1], c='b', ls=':')
        axs[1].set_ylabel('1st Derivative')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_title('Gradient Check (C1 matched)')

        # second derivative diagnostic only
        dd_orig = np.gradient(d_orig, self.psi_orig)
        dd_mod = np.gradient(d_mod, self.psi)
        axs[2].plot(self.psi_orig, dd_orig, 'k--', alpha=0.5, label='Original')
        axs[2].plot(self.psi, dd_mod, 'r-', lw=2, label='Modified')
        if hasattr(self, 'trans_pts'):
            axs[2].axvline(self.trans_pts[0], c='g', ls=':')
            axs[2].axvline(self.trans_pts[1], c='b', ls=':')
        axs[2].set_ylabel('2nd Derivative')
        axs[2].set_xlabel('Psi')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_title('Curvature Diagnostic (C2 not enforced)')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python smooth_pedestal_top_c1.py <file>")
        sys.exit(1)

    mod = ProfileModifier(sys.argv[1])

    # --- USER SETTINGS ---
    PSI_START = 0.94
    PSI_END = 0.99

    # local fit for endpoint slope estimation
    HALF_WINDOW = 4
    POLY_DEG = 3

    # 0.0 = preserve endpoint slopes
    # 1.0 = fully flatten toward secant slope
    SMOOTH_STRENGTH = 0.4

    mod.smooth_pedestal_top(
        PSI_START,
        PSI_END,
        half_window=HALF_WINDOW,
        poly_deg=POLY_DEG,
        smooth_strength=SMOOTH_STRENGTH
    )

    mod.plot()

    base, ext = os.path.splitext(sys.argv[1])
    write_prf(f"{base}_pedtop_smooth_c1{ext}", mod.psi, mod.val)