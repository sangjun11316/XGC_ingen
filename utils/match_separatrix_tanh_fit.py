import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import curve_fit
import sys, os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf
except ImportError as e:
    # Mocking read_prf for standalone generation if module is missing
    print(f"Warning: Could not import 'xgc_ingen'. Using mock data reader if file exists. Details: {e}")
    def read_prf(filename):
        if not os.path.exists(filename): raise FileNotFoundError(filename)
        data = np.loadtxt(filename, skiprows=1, max_rows=None)
        if data.shape[1] == 2: return data[:,0], data[:,1]
        return np.linspace(0, 1.0, 100), np.exp(-np.linspace(0, 1.0, 100)**2)

# --- Helper Functions ---
def write_prf(filename, psi, value):
    """ Writes data to a .prf file in TOMMS format """
    try:
        with open(filename, 'w') as f:
            f.write(f"{len(psi)}\n")
            for p, v in zip(psi, value):
                f.write(f"{p:.12E}  {v:.12E}\n")
            f.write("-1\n")
        print(f">> Written to {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")

# --- Main Logic Class ---
class ProfileModifier:
    def __init__(self, filename, exp_filename=None):
        self.filename = filename
        
        try:
            self.psi, self.val = read_prf(filename)
        except Exception as e:
            print(f"Error reading profile: {e}")
            sys.exit(1)
            
        self.psi_orig = self.psi.copy()
        self.val_orig = self.val.copy()
        self.psi_exp = None
        self.val_exp = None
        
        # Load experimental data if provided
        if exp_filename:
            if os.path.exists(exp_filename):
                print(f"Loading experimental data from {exp_filename}...")
                try:
                    self.psi_exp, self.val_exp = read_prf(exp_filename)
                except Exception as e:
                    print(f"Warning: Failed to read experimental file: {e}")

    def match_separatrix_by_shift(self, target_val):
        """
        Shifts the profile horizontally (in psi) so that the value at psi=1.0 
        equals target_val. Preserves gradients exactly, but moves the pedestal location.
        """
        if target_val > np.max(self.val):
            print(f"Error: Target separatrix value ({target_val:.2e}) is higher than profile maximum.")
            return

        # Use inverse interpolation to find current psi at target_val
        idx_search_start = len(self.psi) // 2
        psi_search = self.psi[idx_search_start:]
        val_search = self.val[idx_search_start:]

        if val_search[0] > val_search[-1]: 
            f_inv = interp1d(val_search[::-1], psi_search[::-1], kind='cubic', bounds_error=False, fill_value="extrapolate")
        else:
            f_inv = interp1d(val_search, psi_search, kind='cubic', bounds_error=False, fill_value="extrapolate")

        current_psi_at_target = f_inv(target_val)
        shift = 1.0 - current_psi_at_target
        
        print(f">> Shifting profile to match separatrix target {target_val:.2e}")
        print(f"   Shift delta_psi = {shift:.4f}")

        # Interpolate shifted values
        f_orig = PchipInterpolator(self.psi, self.val)
        psi_shifted_lookup = self.psi - shift
        new_vals = f_orig(psi_shifted_lookup)
        
        # Handle core boundary
        if shift > 0:
            new_vals[psi_shifted_lookup < self.psi[0]] = self.val[0]
            
        self.val = new_vals

    def match_separatrix_by_tanh_fit(self, target_val, fit_range=(0.8, 1.05)):
        """
        Fits a modified tanh function to the edge data (fit_range) with the 
        strict constraint that f(1.0) == target_val.
        Replaces the edge data with this clean, fitted curve.
        """
        print(f">> Fitting constrained Tanh to edge (Target @ Sep = {target_val:.2e})")

        # 1. Select Data for Fitting
        mask = (self.psi >= fit_range[0]) & (self.psi <= fit_range[1])
        x_data = self.psi[mask]
        y_data = self.val[mask]

        if len(x_data) < 5:
            print("Error: Not enough points in fit range.")
            return

        # 2. Define the Constrained Tanh Model
        # General Tanh: y = A * tanh((x_sym - x) / w) + B
        # Constraint: y(1) = target
        # B = target - A * tanh((x_sym - 1) / w)
        # Substitute B back into equation:
        # y = A * [ tanh((x_sym - x)/w) - tanh((x_sym - 1)/w) ] + target
        
        def constrained_model(x, A, x_sym, w):
            return A * (np.tanh((x_sym - x) / w) - np.tanh((x_sym - 1.0) / w)) + target_val

        # 3. Initial Guesses
        # A (Height): Approx (Max - Min) / 2
        # x_sym (Position): Approx 0.98
        # w (Width): Approx 0.05
        p0 = [(np.max(y_data) - target_val)/2, 0.98, 0.05]
        
        # Bounds to keep fit physical (A>0, 0.9<x_sym<1.1, w>0)
        bounds = ([0, 0.9, 0.001], [np.inf, 1.1, 0.5])

        try:
            popt, _ = curve_fit(constrained_model, x_data, y_data, p0=p0, bounds=bounds, maxfev=5000)
            A_opt, x_sym_opt, w_opt = popt
            print(f"   Fit Params: A={A_opt:.2e}, x_sym={x_sym_opt:.3f}, w={w_opt:.3f}")
        except Exception as e:
            print(f"   Fit failed: {e}")
            return

        # 4. Generate fitted curve & Stitch
        # We replace data starting from fit_range[0]
        # To avoid a jump discontinuity at the start of the replacement, we use a simple blending weight
        
        # Generate clean tanh on the full grid
        val_fitted = constrained_model(self.psi, *popt)

        # Create Blend Mask
        # 1.0 before fit_start, 0.0 after fit_start + blend_width
        blend_start = fit_range[0]
        blend_width = 0.04
        
        # Sigmoid blending function
        # w goes from 0 (original) to 1 (fitted)
        k_blend = 100 # sharpness
        blend_weight = 1.0 / (1.0 + np.exp(-k_blend * (self.psi - (blend_start + blend_width/2))))
        
        # Apply replacement only where psi > blend_start
        mask_replace = self.psi > blend_start
        
        # Blend: (1 - w)*Original + w*Fitted
        self.val[mask_replace] = (1 - blend_weight[mask_replace]) * self.val[mask_replace] + \
                                 blend_weight[mask_replace] * val_fitted[mask_replace]

        # Ensure exact match at boundary 1.0 (though fit guarantees it, blending might smooth it slightly if blend region overlaps)
        # Since blend is usually well before 1.0, the value at 1.0 should be purely the fitted value.

    def double_exponential_sol(self, psi_start=0.9, psi_mid=0.95, val_mid=None, psi_max=1.3, target_floor=10.0):
        """ 
        Creates a two-stage decay for the SOL.
        """
        # --- 1. Identify Core Limit ---
        idx_start = np.argmin(np.abs(self.psi - psi_start))
        psi_core = self.psi[:idx_start+1]
        val_core = self.val[:idx_start+1]
        
        val_start_actual = val_core[-1]
        psi_start_actual = psi_core[-1]

        if val_mid is None: val_mid = val_start_actual * 0.5

        print(f">> Applying Two-Stage Decay (SOL)")
        
        # --- 2. Region 1 ---
        delta_psi_1 = psi_mid - psi_start_actual
        if delta_psi_1 <= 0: delta_psi_1 = 1e-6
        k1 = np.log(val_mid / val_start_actual) / delta_psi_1
        
        n_pts_1 = 30
        psi_r1 = np.linspace(psi_start_actual, psi_mid, n_pts_1 + 1)[1:] 
        val_r1 = val_start_actual * np.exp(k1 * (psi_r1 - psi_start_actual))

        # --- 3. Region 2 ---
        slope_at_mid = val_mid * k1
        decay_amp = val_mid - target_floor
        
        if decay_amp <= 0:
            k2 = 0
            decay_amp = 0
        else:
            k2 = slope_at_mid / decay_amp

        n_pts_2 = 50
        psi_r2 = np.linspace(psi_mid, psi_max, n_pts_2 + 1)[1:]
        val_r2 = decay_amp * np.exp(k2 * (psi_r2 - psi_mid)) + target_floor

        # --- 4. Stitch & Interpolate ---
        psi_knots = np.concatenate([psi_core, psi_r1, psi_r2])
        val_knots = np.concatenate([val_core, val_r1, val_r2])

        dpsi_avg = np.mean(np.diff(self.psi_orig))
        n_new_points = int((psi_max - self.psi_orig[0]) / dpsi_avg)
        self.psi = np.linspace(self.psi_orig[0], psi_max, n_new_points)

        f_interp = interp1d(np.sqrt(psi_knots), val_knots, kind='cubic', bounds_error=False, fill_value="extrapolate")
        self.val = f_interp(np.sqrt(self.psi))
        self.val = np.maximum(self.val, np.min(val_r2))
        self.trans_pts = [psi_start_actual, psi_mid]

    def plot_comparison(self):
        # Create 2 subplots sharing the x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Plot 1: Value ---
        if self.psi_exp is not None:
            ax1.scatter(self.psi_exp, self.val_exp, c='blue', s=30, label='Experimental', alpha=0.4, zorder=1)

        ax1.plot(self.psi_orig, self.val_orig, 'k--', label='Original', alpha=0.6, zorder=2)
        ax1.plot(self.psi,      self.val,      'r-',  label='Modified', linewidth=2, zorder=3)
        
        if hasattr(self, 'trans_pts'):
            ax1.axvline(self.trans_pts[0], color='g', linestyle=':', label='Start Decay 1')
            ax1.axvline(self.trans_pts[1], color='m', linestyle=':', label='Start Decay 2')
        
        ax1.axvline(1.0, color='k', linestyle='-', lw=1, alpha=0.3, label='Separatrix')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Profile Modification: {os.path.basename(self.filename)}')
        ax1.legend()
        ax1.grid(which='major', alpha=0.5)
        #ax1.set_yscale('log')

        # --- Plot 2: Gradient (dValue/dPsi) ---
        # Calculate gradients
        grad_orig = np.gradient(self.val_orig, self.psi_orig)
        grad_mod = np.gradient(self.val, self.psi)
        
        ax2.plot(self.psi_orig, grad_orig, 'k--', label='Original Grad', alpha=0.6)
        ax2.plot(self.psi,      grad_mod,  'r-',  label='Modified Grad', linewidth=2)
        
        # Mark transitions on gradient plot too
        if hasattr(self, 'trans_pts'):
            ax2.axvline(self.trans_pts[0], color='g', linestyle=':', alpha=0.5)
            ax2.axvline(self.trans_pts[1], color='m', linestyle=':', alpha=0.5)
        ax2.axvline(1.0, color='k', linestyle='-', lw=1, alpha=0.3)

        ax2.set_ylabel(r'Gradient ($dVal/d\psi$)')
        ax2.set_xlabel(r'$\psi_N$')
        ax2.grid(which='major', alpha=0.5)
        ax2.set_title("Gradient Check (Smoothness)")

        plt.tight_layout()
        plt.show()

# --- Main Script ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_profile_2stage.py <prf_file> [optional: <exp_prf_file>]")
        sys.exit(1)

    prf_file = sys.argv[1]
    exp_file = sys.argv[2] if len(sys.argv) > 2 else None

    mod = ProfileModifier(prf_file, exp_filename=exp_file)
    
    # --- CONFIGURATION ---
    
    TARGET_SEP_VAL = 100.0   # The value you want at psi = 1.0
    
    # METHOD 1: Shift (Uncomment to use)
    # mod.match_separatrix_by_shift(TARGET_SEP_VAL)
    
    # METHOD 2: Constrained Tanh Fit (Active)
    # Fits a clean tanh from psi=0.85 outwards, forcing it to pass through TARGET_SEP_VAL at psi=1.0
    mod.match_separatrix_by_tanh_fit(TARGET_SEP_VAL, fit_range=(0.92, 1.05))

    # --- SOL DECAY (Applied after the shift/fit) ---
    psi_start = 1.0  # Start decay exactly at separatrix (since we just fixed it there)
    psi_knee  = 1.05
    
    # Use the target value we just set as the starting point for decay logic
    val_at_sep = TARGET_SEP_VAL 
    
    target_val_at_knee = val_at_sep * 0.2  # Fast drop in near-SOL
    floor_val = val_at_sep * 0.01          # Far SOL floor
    
    mod.double_exponential_sol(
        psi_start=psi_start, 
        psi_mid=psi_knee, 
        val_mid=target_val_at_knee, 
        psi_max=1.3, 
        target_floor=floor_val
    )

    mod.plot_comparison()

    base, ext = os.path.splitext(prf_file)
    new_filename = f"{base}_tanhfit{ext}"
    write_prf(new_filename, mod.psi, mod.val)
