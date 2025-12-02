import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf, Eqdsk
except ImportError as e:
    print(f"Error: Could not import 'xgc_ingen' from parent directory.\nDetails: {e}")
    sys.exit(1)

def calc_polygon_metrics(R, Z):
    """
    Calculates Area and Volume of revolution (Torus) for a closed polygon (R, Z).
    Uses Shoelace formula for Area and Pappus's 2nd Theorem for Volume.
    """
    # --- Shoelace Formula ---
    term_cross = R * np.roll(Z, -1) - np.roll(R, -1) * Z
    area_sum = np.sum(term_cross)
    area = 0.5 * np.abs(area_sum)

    # --- Pappus's Theorem ---
    if area > 1e-12:
        term_sum_R = R + np.roll(R, -1)
        centroid_sum = np.sum(term_sum_R * term_cross)
        R_centroid = np.abs(centroid_sum) / (6.0 * area)
        volume = 2.0 * np.pi * R_centroid * area
    else:
        volume = 0.0
    
    return volume

def calculate_volume_profile(eq, psi_grid):
    """
    Calculates the enclosed volume V(psi) for each psi value in psi_grid.
    Returns: V_array (same shape as psi_grid)
    """
    print(f"   Calculating enclosed volumes for {len(psi_grid)} surfaces...")
    
    # Use a hidden figure to extract contours
    fig_temp = plt.figure(figsize=(1,1))
    ax_temp = fig_temp.add_subplot(111)
    
    vol_profile = np.zeros_like(psi_grid)
    
    try:
        for i, level in enumerate(psi_grid):
            # Volume is 0 at or below axis
            if level <= 0.0:
                vol_profile[i] = 0.0
                continue
            
            # Matplotlib contouring
            cntr = ax_temp.contour(eq.r, eq.z, eq.psinrz.T, levels=[level])
            paths = cntr.allsegs[0] 
            
            if len(paths) > 0:
                # Find path with max length (main plasma)
                main_path = max(paths, key=lambda p: len(p))
                R_cont = main_path[:, 0]
                Z_cont = main_path[:, 1]
                V = calc_polygon_metrics(R_cont, Z_cont)
                vol_profile[i] = V
            else:
                vol_profile[i] = 0.0 
                
    except Exception as e:
        print(f"Error in volume calculation: {e}")
    finally:
        plt.close(fig_temp) # Clean up
        
    return vol_profile

def write_output(filename, psi, val):
    try:
        with open(filename, 'w') as f:
            f.write(f"{len(psi)}\n")
            for p, v in zip(psi, val):
                f.write(f"{p:.12E}  {v:.12E}\n")
            f.write("-1\n")
        print(f">> Written result to {filename}")
    except Exception as e:
        print(f"Error writing file {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_prad_to_power.py <prad_file> <g_file>")
        print("Example: python convert_prad_to_power.py inputs/prad.prf inputs/g12345.00000")
        sys.exit(1)

    file_g    = sys.argv[1]
    file_prad = sys.argv[2]

    # 1. Load Data
    print(f"\n>> Loading Prad profile: {file_prad}")
    psi_in, prad_in = read_prf(file_prad, 'Prad') # [MW/m^3]
    
    # Sort input just in case
    sort_idx = np.argsort(psi_in)
    psi_in   = psi_in[sort_idx]
    prad_in  = prad_in[sort_idx]

    print(f">> Loading Equilibrium: {file_g}")
    eq = Eqdsk(file_g)

    # 2. Create Uniform Grid for Integration
    #    Using a dense uniform grid reduces trapezoidal error
    N_UNIFORM = 20
    print(f">> Interpolating onto uniform grid with {N_UNIFORM} points...")
    psi_unif = np.linspace(0, 0.9999, N_UNIFORM)
    
    # Interpolate Prad
    f_prad = interp1d(psi_in, prad_in, kind='linear', fill_value="extrapolate")
    prad_unif = f_prad(psi_unif)
    prad_unif[prad_unif < 0] = 0.0 # Clamp negative noise

    # 3. Calculate Volume Profile V(psi) on Uniform Grid
    psi_vol_calc = np.copy(psi_unif)
    vol_calc = calculate_volume_profile(eq, psi_vol_calc)
    
    # Interpolate Volume to fine uniform grid
    f_vol = interp1d(psi_vol_calc, vol_calc, kind='cubic', fill_value="extrapolate")
    vol_unif = f_vol(psi_unif)
    
    # Ensure V=0 at axis
    vol_unif[psi_unif <= 0] = 0.0

    # 4. Integrate Power
    #    P_cum[i] = Integral (P_dens * dV)
    from scipy.integrate import cumulative_trapezoid
    
    # Integrate Prad against Volume
    p_cum_unif = cumulative_trapezoid(prad_unif, vol_unif, initial=0)

    # 5. Calculate Segment Power (Differential)
    p_seg_unif = np.diff(p_cum_unif, prepend=0.0)

    # 6. Plot Verification
    fig, axes = plt.subplots(1, 4, figsize=(5*4, 5), constrained_layout=True)
    
    # Input Density (Compare Original vs Uniform)
    ax = axes[0]
    ax.plot(psi_in, prad_in, 'k.', markersize=2, label='Input (Raw)', zorder=3)
    ax.plot(psi_unif, prad_unif, 'r.-', label='Uniform Interp')
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(r'Power Density [$MW/m^3$]')
    ax.set_title('Radiated Power Density')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Cumulative Volume
    ax = axes[1]
    ax.plot(psi_unif, vol_unif, 'k.-', label='Enclosed Volume')
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(r'Volume [$m^3$]')
    ax.set_title(f'Enclosed Volume\nTotal = {vol_unif[-1]:.4f} m3')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Cumulative Power (Integral)
    ax = axes[2]
    ax.plot(psi_unif, p_cum_unif, 'b.-', label='Cumulative Power')
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(r'Power [$MW$]')
    ax.set_title(f'Cumulative Power\nTotal = {p_cum_unif[-1]:.4f} MW')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Segment Power (Differential)
    ax = axes[3]
    ax.plot(psi_unif, p_seg_unif, 'g.-', label='Segment Power')
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel(r'Power per Shell [$MW$]')
    ax.set_title(f'Segment Power ($\Delta \psi_N \\approx {1/N_UNIFORM:.1e}$)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.show()

    # 7. Write Outputs (On Uniform Grid)
    base, ext = os.path.splitext(file_prad)
    
    # Save Cumulative Power
    write_output(f"{base}_MW_cum{ext}", psi_unif, p_cum_unif)
    
    # Save Segment Power
    write_output(f"{base}_MW_seg{ext}", psi_unif, p_seg_unif)
    
    # Save Volume (for verification)
    write_output(f"{base}_vol{ext}", psi_unif, vol_unif)
