import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
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

class ProfileModifier:
    def __init__(self, filename):
        self.filename = filename
        self.psi, self.val = read_prf(filename)
        self.psi_orig = self.psi.copy()
        self.val_orig = self.val.copy()

    def _get_start_conditions(self, psi_start):
        idx_start = np.argmin(np.abs(self.psi - psi_start))
        window = 3
        s_slice = slice(max(0, idx_start-window), min(len(self.psi), idx_start+window+1))
        p_local = self.psi[s_slice]
        v_local = self.val[s_slice]
        poly = np.polyfit(p_local, v_local, 1)
        return poly[0], np.polyval(poly, psi_start) # grad, val

    def _generate_modified_grid_and_values(self, psi_start, psi_sep, psi_max, func_r1, func_r2):
        """
        Generates values on the ORIGINAL grid resolution, extending it if necessary.
        """
        # 1. Define Core region (Preserve original data)
        mask_core = self.psi_orig < psi_start
        psi_core = self.psi_orig[mask_core]
        val_core = self.val_orig[mask_core]

        # 2. Define Modified Region Grid (Preserve original points + Extend)
        # Get original points that are in the modified region
        psi_existing_tail = self.psi_orig[~mask_core]

        # Determine if we need to extend the grid to psi_max
        if len(psi_existing_tail) > 0:
            last_psi = psi_existing_tail[-1]
        else:
            last_psi = psi_start

        if last_psi < psi_max:
            # Calculate local dpsi from the last few points to maintain resolution
            if len(self.psi_orig) > 5:
                dpsi_tail = np.mean(np.diff(self.psi_orig)[-5:])
            else:
                dpsi_tail = 0.001
            
            # Generate new points
            psi_extension = np.arange(last_psi + dpsi_tail, psi_max, dpsi_tail)
            psi_mod = np.concatenate([psi_existing_tail, psi_extension])
        else:
            psi_mod = psi_existing_tail

        # 3. Evaluate Analytical Functions on the Grid
        val_mod = np.zeros_like(psi_mod)
        
        # Region 1: psi_start <= psi <= psi_sep
        mask_r1 = (psi_mod <= psi_sep)
        val_mod[mask_r1] = func_r1(psi_mod[mask_r1])
        
        # Region 2: psi > psi_sep
        mask_r2 = (psi_mod > psi_sep)
        val_mod[mask_r2] = func_r2(psi_mod[mask_r2])

        # 4. Stitch
        self.psi = np.concatenate([psi_core, psi_mod])
        self.val = np.concatenate([val_core, val_mod])

    def solve_smooth_tanh_connection(self, psi_start, target_val_sep, floor_val, psi_sep=1.0, w_tanh=0.04):
        """
        Solves for a smooth Tanh connection preserving grid resolution.
        """
        grad_start, val_start = self._get_start_conditions(psi_start)
        print(f">> Solving Tanh Connection (Preserving Grid):")
        print(f"   Start: psi={psi_start:.3f}, val={val_start:.3e}, grad={grad_start:.3e}")

        # --- Solve Parameters ---
        def equations(x_sym):
            z_st  = (x_sym - psi_start) / w_tanh
            z_sep = (x_sym - psi_sep)   / w_tanh
            sech2_st = 1.0 - np.tanh(z_st)**2
            if abs(sech2_st) < 1e-10: return 1e10
            
            A_implied = -grad_start * w_tanh / sech2_st
            val_diff_calc = A_implied * (np.tanh(z_st) - np.tanh(z_sep))
            val_diff_target = val_start - target_val_sep
            return val_diff_calc - val_diff_target

        x_sym_guess = (psi_start + psi_sep) / 2.0
        x_sym_sol, info, ier, msg = fsolve(equations, x_sym_guess, full_output=True)
        x_sym = x_sym_sol[0] if ier == 1 else x_sym_guess

        z_st = (x_sym - psi_start) / w_tanh
        sech2_st = 1.0 - np.tanh(z_st)**2
        A = -grad_start * w_tanh / sech2_st
        C = val_start - A * np.tanh(z_st)

        print(f"   [Tanh Params] x_sym={x_sym:.4f}, A={A:.2e}, C={C:.2e}")

        # --- Region 2 Params ---
        z_sep = (x_sym - psi_sep) / w_tanh
        sech2_sep = 1.0 - np.tanh(z_sep)**2
        grad_at_sep = - (A / w_tanh) * sech2_sep
        
        amp_sol = target_val_sep - floor_val
        lam = 0.05 if (abs(grad_at_sep) < 1e-12 or amp_sol <= 0) else -amp_sol / grad_at_sep
        print(f"   [SOL Decay] Lambda={lam:.4f}")

        # --- Generate Grid & Eval ---
        # Define the lambda functions for the two regions
        func_r1 = lambda p: A * np.tanh((x_sym - p)/w_tanh) + C
        func_r2 = lambda p: amp_sol * np.exp(-(p - psi_sep)/lam) + floor_val
        
        self.trans_pts = [psi_start, psi_sep]
        self._generate_modified_grid_and_values(psi_start, psi_sep, 1.3, func_r1, func_r2)


    def solve_smooth_connection(self, psi_start, target_val_sep, floor_val, psi_sep=1.0, k_shape=None):
        """
        Solves for Exponential connection preserving grid resolution.
        """
        grad_start, val_start = self._get_start_conditions(psi_start)
        print(f">> Solving Exp Connection (Preserving Grid):")
        dx = psi_sep - psi_start
        
        # --- Solve Parameters ---
        if target_val_sep is not None:
            dVal = target_val_sep - val_start
            def eq_k(k):
                if abs(grad_start) < 1e-9: return 1e9 
                return np.exp(k * dx) - 1.0 - (k * dVal / grad_start)
            k_sol, _, ier, _ = fsolve(eq_k, -5.0, full_output=True)
            k = k_sol[0] if ier == 1 else -10.0
        else:
            k = k_shape if k_shape is not None else 30.0

        if abs(k) < 1e-9: k = 1e-9
        A = grad_start / k
        C = val_start - A
        
        if target_val_sep is None:
            target_val_sep = A * np.exp(k * dx) + C

        grad_at_sep = A * k * np.exp(k * dx)
        amp_sol = target_val_sep - floor_val
        lam = 0.05 if (abs(grad_at_sep) < 1e-12 or amp_sol <= 0) else -amp_sol / grad_at_sep
            
        # --- Generate Grid & Eval ---
        func_r1 = lambda p: A * np.exp(k * (p - psi_start)) + C
        func_r2 = lambda p: amp_sol * np.exp(-(p - psi_sep)/lam) + floor_val
        
        self.trans_pts = [psi_start, psi_sep]
        self._generate_modified_grid_and_values(psi_start, psi_sep, 1.3, func_r1, func_r2)

    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        axs[0].plot(self.psi_orig, self.val_orig, 'k--', label='Original', alpha=0.5)
        axs[0].plot(self.psi, self.val, 'r-', lw=2, label='Modified')
        if hasattr(self, 'trans_pts'):
            axs[0].axvline(self.trans_pts[0], c='g', ls=':', label='Start')
            axs[0].axvline(self.trans_pts[1], c='b', ls=':', label='Separatrix')
        axs[0].set_ylabel('Value')
        axs[0].set_yscale('log')
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
        
        d_orig = np.gradient(self.val_orig, self.psi_orig)
        d_mod  = np.gradient(self.val, self.psi)
        axs[1].plot(self.psi_orig, d_orig, 'k--', alpha=0.5)
        axs[1].plot(self.psi, d_mod, 'r-', lw=2)
        axs[1].set_ylabel('Gradient')
        axs[1].set_xlabel('Psi')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_title('Gradient Check')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python modify_profile_smooth_solver.py <file>")
        sys.exit(1)
        
    mod = ProfileModifier(sys.argv[1])
    
    # --- USER SETTINGS ---
    PSI_START = 0.94   
    PSI_SEP   = 1.00   
    FLOOR_VAL = 2.38945E17  

    # --- CHOOSE MODE ---
    
    # Option 1: Exponential
    # mod.solve_smooth_connection(PSI_START, target_val_sep=None, floor_val=FLOOR_VAL, psi_sep=PSI_SEP, k_shape=40.0)

    # Option 2: Tanh (Pedestal Shape)
    TARGET_SEP = 0.6E18
    WIDTH      = 0.04
    mod.solve_smooth_tanh_connection(PSI_START, TARGET_SEP, FLOOR_VAL, PSI_SEP, w_tanh=WIDTH)
    
    mod.plot()
    
    base, ext = os.path.splitext(sys.argv[1])
    write_prf(f"{base}_tanh_smooth{ext}", mod.psi, mod.val)
