import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys
import os

# --- Import from parent directory ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from xgc_ingen import read_prf
except ImportError as e:
    print(f"Error: Could not import 'xgc_ingen' from parent directory.\nDetails: {e}")
    sys.exit(1)

def calculate_Zimp(file_ne, file_nimp, file_Zeff):
    """
    Calculates Z_imp based on:
      1) Quasi-neutrality: ne = nD*1 + nimp*Zimp
      2) Zeff def:         ne*Zeff = nD*1^2 + nimp*Zimp^2
    
    Returns:
      psi, Zimp, nD, ne, nimp, Zeff
    """
    print(f"\n>> Calculating Z_imp...")
    # 1. Read Data
    psi_ne, ne = read_prf(file_ne, 'ne')
    psi_nimp, nimp = read_prf(file_nimp, 'n_imp')
    psi_Zeff, Zeff = read_prf(file_Zeff, 'Z_eff')

    # 2. Interpolate onto the master grid (using psi_ne)
    #    We use 'ne' grid as master because it defines the plasma density.
    f_nimp = interp1d(psi_nimp, nimp, kind='linear', bounds_error=False, fill_value="extrapolate")
    f_Zeff = interp1d(psi_Zeff, Zeff, kind='linear', bounds_error=False, fill_value="extrapolate")

    nimp_interp = f_nimp(psi_ne)
    Zeff_interp = f_Zeff(psi_ne)

    # Plots
    # ne
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Input Data Verification')
    
    ax[0].plot(psi_ne, ne, 'k.-', label='Input')
    ax[0].set_title('Electron Density ($n_e$)')
    ax[0].set_xlabel(r'$\psi_N$')
    ax[0].set_ylabel('$m^{-3}$')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # nimp (Original vs Interp)
    ax[1].plot(psi_nimp, nimp, 'kx', label='Original Data')
    ax[1].plot(psi_ne, nimp_interp, 'r-', label='Interpolated', alpha=0.7)
    ax[1].set_title('Impurity Density ($n_{imp}$)')
    ax[1].set_xlabel(r'$\psi_N$')
    ax[1].set_ylabel('$m^{-3}$')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    # Zeff (Original vs Interp)
    ax[2].plot(psi_Zeff, Zeff, 'kx', label='Original Data')
    ax[2].plot(psi_ne, Zeff_interp, 'b-', label='Interpolated', alpha=0.7)
    ax[2].set_title('Effective Charge ($Z_{eff}$)')
    ax[2].set_xlabel(r'$\psi_N$')
    ax[2].grid(True, alpha=0.3)
    ax[2].legend()

    plt.tight_layout()
    plt.show()

    # 3. Solve Quadratic Equation for Zimp
    # From the system of equations, we derived:
    # Zimp^2 - Zimp - (ne/nimp)*(Zeff - 1) = 0
    ratio_ne_nimp = ne / nimp_interp
    C = ratio_ne_nimp * (Zeff_interp - 1.0)
    Zimp = (1.0 + np.sqrt(1.0 + 4.0*C)) / 2.0

    # 4. Calculate nD (Deuterium Density)
    nD = ne - Zimp * nimp_interp

    return psi_ne, ne, nD, nimp_interp, Zimp, Zeff_interp

def sanity_check(psi, ne, nD, nimp, Zeff, Zimp):
    print("\n>> Performing Sanity Check...")
    
    # 1. Check Quasi-neutrality: ne = nD + Zimp*nimp
    ne_recon = nD * 1.0 + nimp * Zimp
    err_qn = np.abs(ne - ne_recon)
    max_err_qn = np.max(err_qn)
    
    # 2. Check Zeff: Zeff = (nD + nimp*Zimp^2) / ne
    # Avoid div by zero
    Zeff_recon = (nD * 1.0**2 + nimp * Zimp**2) / ne
    err_Zeff = np.abs(Zeff - Zeff_recon)
    max_err_Zeff = np.max(err_Zeff)

    print(f"   Max Error (Quasi-neutrality): {max_err_qn:.4e}")
    print(f"   Max Error (Zeff reconstruction): {max_err_Zeff:.4e}")

    if max_err_qn > 1e-10 or max_err_Zeff > 1e-10:
        print("   [WARNING] Sanity check failed! Errors are large.")
    else:
        print("   [PASS] Sanity check passed.")

def plot_results(psi, ne, nD, nimp, Zeff, Zimp):
    #--- Plot profiles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Densities
    ax = axes[0]
    ax.plot(psi, ne, 'k-', label=r'$n_e$', lw=2)
    ax.plot(psi, nD, 'b--', label=r'$n_D$ (derived)')
    # Plot n_imp * Z_imp to compare charge contribution
    ax.plot(psi, nimp * Zimp, 'r:', label=r'$n_{imp} \cdot Z_{imp}$')
    ax.plot(psi, nD + nimp*Zimp, c='gray', ls=':', label=r'$n_D + n_{imp} \cdot Z_{imp}$')    

    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel('Density [$m^{-3}$]')
    ax.set_title('Densities')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Charge States
    ax = axes[1]
    ax.plot(psi, Zimp, 'm-', label=r'$Z_{imp}$ (derived)', lw=2)
    ax.plot(psi, Zeff, 'k-', label=r'$Z_{eff}$ (input)')
    ax.plot(psi, (nD + nimp*Zimp**2)/ne, c='gray', ls=':', label=r'$n_D + n_{imp} \cdot Z_{imp}^2$')
    
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel('Charge')
    ax.set_title(r'Effective Charge & Derived $Z_{imp}$')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    #--- Plot errors
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Densities
    ax = axes[0]
    ax.plot(psi, ne - (nD + nimp*Zimp), c='k', ls='-')    

    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel('Density [$m^{-3}$]')
    ax.set_title('Density Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Charge States
    ax = axes[1]
    ax.plot(psi, Zeff - (nD + nimp*Zimp**2)/ne, c='k', ls='-')
    
    ax.set_xlabel(r'$\psi_N$')
    ax.set_ylabel('Charge')
    ax.set_title(r'Zeff Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


    plt.show()

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
    if len(sys.argv) < 4:
        print("Usage: python calculate_Z_imp.py <file_ne> <file_nimp> <file_Zeff>")
        print("Example: python calculate_Z_imp.py inputs/ne.prf inputs/nimp.prf inputs/Zeff.prf")
        sys.exit(1)

    file_ne   = sys.argv[1]
    file_nimp = sys.argv[2]
    file_Zeff = sys.argv[3]

    # Calculate
    psi, ne, nD, nimp, Zimp, Zeff = calculate_Zimp(file_ne, file_nimp, file_Zeff)
    
    # Check
    sanity_check(psi, ne, nD, nimp, Zeff, Zimp)
    
    # Plot
    plot_results(psi, ne, nD, nimp, Zeff, Zimp)

    # Write Output (Zimp and nD)
    write_output("output_Zimp.prf", psi, Zimp)
    write_output("output_ni_main.prf", psi, nD)
