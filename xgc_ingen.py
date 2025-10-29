import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg') # to enable interactive backend
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline # RectBivariateSpline is often better than interp2d
from pathlib import Path

PROTON_MASS   = 1.6720e-27  # kg
ELECTRON_MASS = 9.1094e-31  # kg
UNIT_CHARGE   = 1.6022e-19  # C

# some general helper functions (TODO: make a separate module)
def is_monotonic(arr):
  diffs = np.diff(arr)
  return np.all(diffs >= 0) or np.all(diffs <= 0)

class Eqdsk:
    def __init__(self, filename):
        self.filename = filename
        print(f">> load g-file: {self.filename}")

        self._read_gfile()

    def _read_1d(self, f, num, width=16):
        """ Helper to read 1D array, correctly handling multiple lines """
        data = np.zeros(num)
        items_read = 0
        # Loop until the required number of items is read
        while items_read < num:
            line = f.readline()
            if not line: # Check for unexpected end of file
                raise EOFError(f"Expected {num} values, but reached end of file after reading {items_read}.")
            # remove newline characters
            line = line.rstrip('\n')
            # calculate how many full items are on this line
            items_on_line = len(line) // width
            # read items from the current line
            for i in range(items_on_line):
                if items_read < num:
                    start_index = i * width
                    end_index = start_index + width
                    try:
                        data[items_read] = float(line[start_index:end_index])
                    except ValueError:
                        raise ValueError(f"Could not convert '{line[start_index:end_index]}' to float. Line: '{line}'")
                    items_read += 1
                else:
                    break 
        return data

    def _read_2d(self, f, nrow, ncol, width=16):
        """
        Helper to read a 2D array (nrow x ncol) of fixed-width floats
        from a file object, correctly handling multiple lines.
        """
        total_num = nrow * ncol
        data = self._read_1d(f, total_num, width)
        data = data.reshape((nrow, ncol), order='F')
    
        return data

    def _read_gfile(self):
        with open(self.filename, 'r') as f:
            # line 1: header and dimensions
            line_str = f.readline()
            line_parts = line_str.split()
            self.header = line_str[:line_str.rfind(line_parts[-3])].strip() # Get text part
            self.idum, self.nw, self.nh = map(int, line_parts[-3:]) # mw, mh

            # line 2: grid geometry
            line_vals = self._read_1d(f, 5)
            self.rdim, self.zdim, self.rcentr, self.rleft, self.zmid = line_vals

            # line 3: magnetic axis, boundary psi, bcenter
            line_vals = self._read_1d(f, 5)
            self.rmag, self.zmag, self.smag, self.sbdy, self.bcentr = line_vals

            # line 4: plasma current, etc.
            line_vals = self._read_1d(f, 5)
            self.ip, self.smag_check, _, self.rmag_check, _ = line_vals 

            # line 5: Z-axis etc. (often duplicates)
            line_vals = self._read_1d(f, 5)
            self.zmag_check, _, self.sbdy_check, _, _ = line_vals

            # 1D profiles
            self.fpol  = self._read_1d(f, self.nw)
            self.pres  = self._read_1d(f, self.nw)
            self.ffp   = self._read_1d(f, self.nw) # F*F'
            self.pp    = self._read_1d(f, self.nw) # P' 
            self.psirz = self._read_2d(f, self.nw, self.nh) # [nw, nh]
            self.q     = self._read_1d(f, self.nw) # q profile

            # boundaries
            line_part = f.readline().split()
            self.nsep, self.nlim = map(int, line_part[:2])
            self.rzsep = self._read_2d(f, 2, self.nsep).T # [nsep, 2]
            self.rzlim = self._read_2d(f, 2, self.nlim).T # [nlim, 2]

        # basic grid setup
        self.r = np.linspace(self.rleft, self.rleft+self.rdim, self.nw)
        self.z = np.linspace(self.zmid-self.zdim/2,  self.zmid+self.zdim/2, self.nh)
        self.dr = self.r[1] - self.r[0]
        self.dz = self.z[1] - self.z[0]

        # normalized psi
        self.psinrz = (self.psirz - self.smag) / (self.sbdy - self.smag)
        self.psin = np.linspace(0, 1, self.nw)

        # etc
        self.bmag = self.fpol[0]/self.rmag

        self.construct_area_volume()

        return

    def construct_area_volume(self):
        print(">> construct area and volume")
        if not hasattr(self, 'rzsep'):
            print("Warning: No rzsep. Equilibrium data might not be fully processed.")
            return

        # Extract R and Z coordinates of the separatrix polygon vertices
        R = self.rzsep[:, 0]
        Z = self.rzsep[:, 1]
        n_points = len(R)

        # --- Calculate Area using Shoelace Formula ---
        # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
        # Sum over i: (R[i] * Z[i+1] - R[i+1] * Z[i])
        # Use np.roll to handle the wrap-around (Z[n+1] -> Z[0])
        area_sum = np.sum(R * np.roll(Z, -1) - np.roll(R, -1) * Z)
        self.area = 0.5 * np.abs(area_sum)

        # --- Calculate Volume using Pappus's Theorem ---
        # Volume = 2 * pi * R_centroid * Area
        # R_centroid = (1 / (6 * Area)) * Sum[(R[i] + R[i+1]) * (R[i] * Z[i+1] - R[i+1] * Z[i])]
        centroid_sum = np.sum((R + np.roll(R, -1)) * (R * np.roll(Z, -1) - np.roll(R, -1) * Z))

        if self.area > 1e-12: # Avoid division by zero for degenerate polygons
            R_centroid = np.abs(centroid_sum) / (6.0 * self.area)
            self.volume = 2.0 * np.pi * R_centroid * self.area
        else:
            print("   Warning: Calculated area is close to zero. Volume calculation skipped.")
            R_centroid = np.mean(R) # Fallback, might not be accurate
            self.volume = 0.0

        return self.area, self.volume

    def plot_overview(self):
        if not hasattr(self, 'psinrz'):
            print("g-file not read or processed.")
            return

        print("----------------------------------")
        buffer = f'Info:\n{self.header}'

        buffer += '\n\n'
        buffer += '{:16s} {:7.3f} [T] \n'.format('Center field',self.bcentr)
        buffer += '{:16s} {:7.3f} [MA]\n'.format('Plasma current',self.ip/1.E6)
        buffer += '{:16s} {:7.3f} [m] \n'.format('Magnetic axis R',self.rmag)
        buffer += '{:16s} {:7.3f} [m] \n'.format('Magnetic axis Z',self.zmag)
        buffer += '{:16s} {:7.3f} [T] \n'.format('Toroidal field',self.fpol[0]/self.rmag)
        buffer += '{:16s} {:7.3f} [Wb] \n'.format('Poloidal flux',self.sbdy - self.smag)
        buffer += '{:16s} {:7.3f} [Wb] \n'.format('psi center',self.smag)

        buffer += '{:16s} {:7.3f}     \n'.format('Q0',self.q[0])
        qf = interp1d(self.psin, self.q)
        buffer += '{:16s} {:7.3f}     \n'.format('Q95',qf(0.95))
        buffer += '{:16s} {:7.3f}     \n'.format('Qedge',self.q[-1])

        if self.area:
            buffer += '\n{:16s} {:7.3f} [m2] \n'.format('Area',self.area)
            buffer +=   '{:16s} {:7.3f} [m3] \n'.format('Volume',self.volume)

        buffer += '\nCtr-clockwise direction is (+)'

        print(buffer)
        print("----------------------------------")

        fig,ax = plt.subplots(figsize=(8, 10))
        cntr = ax.contour(self.r, self.z, self.psinrz.T, levels=np.linspace(0, 1.2, 61)) # Transpose psinrz
        fig.colorbar(cntr, ax=ax, label='Normalized Poloidal Flux ($\psi_N$)')
        ax.plot(self.rzsep[:, 0], self.rzsep[:, 1], 'tab:orange', linewidth=2, label='Separatrix')
        ax.plot(self.rzlim[:, 0], self.rzlim[:, 1], 'k', linewidth=2, label='Limiter')
        ax.plot(self.rmag, self.zmag, 'kx', markersize=10, mew=2, label='Magnetic Axis')
        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title(f'Equilibrium Overview ({self.header})')
        ax.axis('equal')
        ax.legend()
        ax.grid(True)
        plt.show()

class TommsInputGenerator:
    def __init__(self):
        # setups
        self.params = self._default_parameters()

        self.eq = None
        self.midplane    = {}
        self.prof        = {}
        self.prof_interp = {}
        self.resolution  = {}
        self.surface     = {}
        self.wall        = {}

        # flags to track progress
        self.equilibrium_loaded    = False
        self.profiles_loaded       = False
        self.midplane_setted       = False
        self.profiles_interpolated = False
        self.resolution_determined = False
        self.surface_generated     = False
        self.wall_generated        = False
        self.files_written         = False

        # interactive loop
        self._run_interface()

    def _default_parameters(self):
        '''
        params = {
            'g_file'  : './input/g184833.04800_new',

            'te_file' : './input/te_d184833_4800_pfile_new.prf',
            'ti_file' : './input/ti_d184833_4800_pfile_adj_08.prf',
            'ne_file' : './input/ne_d184833_4800_pfile_new.prf',
        }
        '''

        params = {
            'g_file'  : './input2/g179444.02277',

            'te_file' : './input2/179444_T_e',
            'ti_file' : './input2/179444_T_12C6',
            'ne_file' : './input2/179444_n_e',
        }

        # ad-hoc to test TCV
        #params['g_file'] = './input3/EQDSK_51392t0.6000_COCOS02_original'

        # general settings
        params['num_mid']        = 1000
        params['dr_scale_psin']  = [0.0, 0.2, 0.3, 0.6, 0.8, 1.0, 1.05, 1.07, 1.1, 1.2]
        params['dr_scale_fac']   = [2.0, 2.0, 1.2, 1.0, 1.0, 1.0, 1.2,  1.5,  4.0, 12.0]
        params['dr_min']         = 1e-3 # m
        params['pol_scale_fac']  = 1.3
        params['pol_core_reduction_fac'] = 1.0 # not sure how useful this is

        # outputs
        params['output_dir']       = './outputs'
        params['output_surf_file'] = 'surf_xgc1.txt'
        params['output_dpol_file'] = 'dpol_xgc1.txt'
        params['output_wall_file'] = 'wallcurve.txt'
    
        return params

    def _read_equilibrium(self):
        try: 
            self.eq = Eqdsk(self.params['g_file'])
            self.equilibrium_loaded = True
        except Exception as e:
            print(f"Error loading equilibrium file: {e}")
            self.equilibrium_loaded = False

        self.eq.plot_overview()

    def _get_midplane_mapping(self):
        if not self.equilibrium_loaded:
            print("Warning: Load equilibrium first.")
            return

        print(">> get midplane mapping")

        # upto limiter
        rmid = np.linspace(self.eq.rmag, np.amax(self.eq.rzlim[:,0]), self.params['num_mid'])
        zmid = np.full_like(rmid, self.eq.zmag)

        '''
        psi_interpolator = RectBivariateSpline(self.eq.r, self.eq.z, self.eq.psirz, kx=3, ky=3)
        psimid = psi_interpolator.ev(rmid, zmid)
        psinmid = (psimid - self.eq.smag) / (self.eq.sbdy - self.eq.smag)
        '''

        psi_interpolator = RectBivariateSpline(self.eq.r, self.eq.z, np.sqrt(self.eq.psirz - self.eq.smag), kx=3, ky=3)
        psimid = (psi_interpolator.ev(rmid, zmid))**2
        psinmid = psimid / (self.eq.sbdy - self.eq.smag)

        # fix the value at the axis to zero
        psinmid[0] = 0.0

        if not is_monotonic(psinmid):
            print("Warning: midplane['psin'] is not monotonic")

        self.midplane['r']    = rmid
        self.midplane['z']    = zmid
        self.midplane['psin'] = psinmid

        self.midplane_setted = True

    def _read_prf(self, filename, prefix=''):
        print(f"...reading {prefix}: {filename}")
        with open(filename, 'r') as file:
            [n] = map(int, file.readline().strip().split())

            psi=np.zeros(n)
            var=np.zeros(n)

            lines_read = 0
            for l in range(n):
                line = file.readline()
                if not line: # Check for unexpected end of file
                    raise EOFError(f"Expected {n} data lines, but file ended after {lines_read} lines.")
                try:
                    psi[l], var[l] = map(float, line.strip().split())
                    lines_read += 1
                except ValueError:
                    raise ValueError(f"Could not parse line {l+2} as two floats: '{line.strip()}'")

            # --- Optionally check end flag ---
            end_flag_line = file.readline()
            if end_flag_line: # Check if a line was actually read
                try:
                    end_flag = int(end_flag_line.strip().split()[0])
                    if end_flag != -1:
                        print(f"Warning: Expected end flag -1 in {filename}, but found {end_flag}.\n...Proceeding anyway.")
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse end flag line in {filename}: '{end_flag_line.strip()}'.\n...Proceeding anyway.")
            else:
                # No end flag line found, which is okay based on your requirement
                print(f"Warning: No end flag line found in {filename} after reading {n} data points.\n...Proceeding anyway.")

            return psi, var

    def _read_profiles(self):
        if not self.equilibrium_loaded:
            print("Warning: Load equilibrium first.")
            return
        try:
            print(">> load te, ti, ne profiles")

            psi_te, te = self._read_prf(self.params['te_file'], 'Te')
            psi_ti, ti = self._read_prf(self.params['ti_file'], 'Ti')
            psi_ne, ne = self._read_prf(self.params['ne_file'], 'ne')
            self.prof = {'psi_te': psi_te, 'te': te,
                         'psi_ti': psi_ti, 'ti': ti,
                         'psi_ne': psi_ne, 'ne': ne}
            self.profiles_loaded = True
        except Exception as e:
            print(f"Error loading profile files: {e}")
            self.profiles_loaded = False        

    def _interpolate_profiles(self):
        if not self.midplane_setted or not self.profiles_loaded:
            print("Warning: Midplane mapping and raw profiles must be loaded first.")
            return

        print(">> interpolate profiles onto midplane grid")

        psi_target = self.midplane['psin']

        # interp1d
        val_first = self.prof['te'][0]
        val_last = self.prof['te'][-1]
        f_te = interp1d(self.prof['psi_te'], self.prof['te'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        val_first = self.prof['ti'][0]
        val_last = self.prof['ti'][-1]
        f_ti = interp1d(self.prof['psi_ti'], self.prof['ti'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        val_first = self.prof['ne'][0]
        val_last = self.prof['ne'][-1]
        f_ne = interp1d(self.prof['psi_ne'], self.prof['ne'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        self.prof_interp['te'] = f_te(psi_target) # eV
        self.prof_interp['ti'] = f_ti(psi_target) # eV
        self.prof_interp['ne'] = f_ne(psi_target) # m-3
        
        # ensure profiles don't go unphysically negative after extrapolation
        #self.prof_interp['te'][self.prof_interp['te'] < 0] = 1e-3
        #self.prof_interp['ti'][self.prof_interp['ti'] < 0] = 1e-3
        #self.prof_interp['ne'][self.prof_interp['ne'] < 0] = 1e-3

        self.profiles_interpolated = True

    def plot_profiles(self):
        if not self.profiles_loaded:
            print("Warning: Profiles to plot have not been loaded yet")
            return

        fig,ax = plt.subplots(figsize=(7,5))

        if self.prof:
            ax.plot(self.prof['psi_te'], self.prof['te']/1E3, label='raw te [keV]')
            ax.plot(self.prof['psi_ti'], self.prof['ti']/1E3, label='raw ti [keV]')
            ax.plot(self.prof['psi_ne'], self.prof['ne']/1E19, label='raw ne [1E19]')

        if self.prof_interp:
            ax.plot(self.midplane['psin'], self.prof_interp['te']/1E3, c='k',  ls='--', label='__interp te [keV]')
            ax.plot(self.midplane['psin'], self.prof_interp['ti']/1E3, c='k',  ls='--', label='__interp ti [keV]')
            ax.plot(self.midplane['psin'], self.prof_interp['ne']/1E19, c='k', ls='--', label='__interp ne [1E19]')

        ax.set_xlabel('Normalized Poloidal Flux ($\psi_N$)')
        ax.set_ylabel('a.u.')
        ax.set_title('Initial Raw Profiles')
        ax.grid(True, alpha=0.5)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _determine_resolutions(self):
        if not self.profiles_interpolated:
            print("Warning: Interpolated profiles needed")
            return

        print(">> determine resolution")

        rmid  = self.midplane['r']
        psin  = self.midplane['psin']
        te_ev = self.prof_interp['te']
        ti_ev = self.prof_interp['ti']
        ne    = self.prof_interp['ne']

        pi = UNIT_CHARGE*ne*ti_ev
        pe = UNIT_CHARGE*ne*te_ev
        ptot = pi + pe

        # deuterium
        mi = 2*PROTON_MASS

        Bmid = np.abs(self.eq.bmag) * self.eq.rmag / rmid
        rhoi = np.sqrt(mi*ti_ev*UNIT_CHARGE)/(UNIT_CHARGE*Bmid) # TODO: check formula

        # scale factors
        val_first = self.params['dr_scale_fac'][0]
        val_last  = self.params['dr_scale_fac'][-1]
        dr_scale_fac_interpolator = interp1d(self.params['dr_scale_psin'], self.params['dr_scale_fac'], kind='linear', bounds_error=False, fill_value=(val_first, val_last))
        dr_scale_fac = dr_scale_fac_interpolator(psin)

        # radial resolution
        dr_target = rhoi * dr_scale_fac
        dr_target[dr_target < self.params['dr_min']] = self.params['dr_min']

        # poloidal resolution
        dpol_target = dr_target * self.params['pol_scale_fac']

        self.resolution['rhoi'] = rhoi
        self.resolution['dr_target']   = dr_target
        self.resolution['dpol_target'] = dpol_target

        self.resolution_determined = True

    def plot_resolutions(self):
        if not self.resolution_determined:
            print("Warning: Resolutions have not been determined yet")
            return

        rmid      = self.midplane['r']
        psin      = self.midplane['psin']
        rhoi      = self.resolution['rhoi']
        dr_target = self.resolution['dr_target']

        fig,ax = plt.subplots(figsize=(6,5))

        ax.plot(psin, rhoi, c='tab:blue', label='rhoi')
        ax.plot(psin, dr_target, c='tab:orange', label='dr_target')

        ax.set_xlabel('Normalized Poloidal Flux ($\psi_N$)')
        ax.set_ylabel('Length Scale [m]')
        ax.set_title('Target Resolution vs. Ion Gyroradius')
        ax.legend()
        ax.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def _generate_surfaces(self):
        if not self.resolution_determined:
            print("Warning: Resolution have not been determined yet")
            return

        print(">> generate surfaces")

        rmid        = self.midplane['r']
        psin        = self.midplane['psin']
        rhoi        = self.resolution['rhoi']
        dr_target   = self.resolution['dr_target']
        dpol_target = self.resolution['dpol_target']

        val_first = dr_target[0]
        val_last  = dr_target[-1]
        dr_interpolator = interp1d(rmid, dr_target, kind='linear', bounds_error=False, fill_value=(val_first, val_last))

        # start from rmid[0]
        rsurf = [rmid[0]]
        nsurf = 1

        # make first distance large
        dr_now = dr_interpolator(rsurf[-1])*2.0

        rsurf.append(rsurf[-1]+dr_now)
        nsurf += 1

        # then accumulate the rest of dr
        while rsurf[-1] < rmid[-1]:
            dr_now = dr_interpolator(rsurf[-1])

            rsurf.append(rsurf[-1]+dr_now)
            nsurf += 1

        rsurf = np.array(rsurf)
        print(f"...{len(rsurf)} surfaces in R (sanity check: nsurf {nsurf})")

        # get corresponding psin
        val_first = np.sqrt(psin[0])
        val_last  = np.sqrt(psin[-1])
        psin_interpolator = interp1d(rmid, np.sqrt(psin), kind='linear', bounds_error=False, fill_value=(val_first, val_last))
        psurf = (psin_interpolator(rsurf))**2

        # ensure first psurf is 0.0
        psurf[0] = 0.0

        # renormalize psurf so that it hits exactly 1.0
        renorm = psurf[np.squeeze(np.where(psurf > 1.0))[0]]
        psurf_renorm = psurf/renorm

        # calculate poloidal distance
        val_first = dpol_target[0]
        val_last  = dpol_target[-1]
        dpol_interpolator = interp1d(rmid, dpol_target, kind='linear', bounds_error=False, fill_value=(val_first, val_last))
        pol_dist = dpol_interpolator(rsurf)

        # put smaller value near axis to avoid error
        pol_dist[0] *= self.params['pol_core_reduction_fac']
        pol_dist[1] *= self.params['pol_core_reduction_fac']

        self.surface['r']    = rsurf
        self.surface['psin'] = psurf_renorm
        self.surface['dpol'] = pol_dist

        self.surface_generated = True

    def plot_surfaces(self):
        if not self.surface_generated:
            print("Warning: Surfaces have not been determined yet")
            return

        rsurf         = self.surface['r']
        psurf_renorm  = self.surface['psin']
        pol_dist     = self.surface['dpol']

        fig,ax = plt.subplots(figsize=(6,5))

        ax.plot(rsurf, label='rsurf')
        ax.plot(psurf_renorm, label='psurf (renorm)')
        ax.plot(pol_dist*1E2, label='pol_dist * 1E2 [cm]')

        ax.set_title('Surfaces')
        ax.legend()
        ax.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    # TODO: generate wall file from extenally given information than included in Eqdsk
    def _generate_wall(self):
        if not self.equilibrium_loaded:
            print("Warning: Load equilibrium first.")
            return
        

    def _write_tomms_input(self):
        if not self.surface_generated:
            print("Warning: Surfaces have not been determined yet")
            return

        print(">> write TOMMS input")

        rsurf         = self.surface['r']
        nsurf         = len(rsurf)
        psurf_renorm  = self.surface['psin']
        pol_dist      = self.surface['dpol']

        output_dir = Path(self.params['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        surf_file  = output_dir / self.params['output_surf_file']
        dpol_file  = output_dir / self.params['output_dpol_file']

        # surface file
        try:
            with open(surf_file, 'w') as f:
                f.write(f"{nsurf}\n")
                for val in psurf_renorm:
                    f.write(f"{val:22.12e}\n")
            print(f"... {surf_file}")
        except Exception as e:
            print(f"Error writing {surf_file}: {e}")
            return       

        # dpol file
        try:
            with open(dpol_file, 'w') as f:
                f.write(f"{nsurf}\n")
                for i in range(nsurf):
                    f.write(f"{psurf_renorm[i]:22.12e}   {pol_dist[i]:22.12e}\n")
            print(f"... {dpol_file}")
            self.files_written = True 
        except Exception as e:
            print(f"Error writing {dpol_file}: {e}")

    def _run_interface(self):
        while True:
            status_eq = 'yes' if self.equilibrium_loaded else 'no'
            print("##################################")
            print("#      TOMMS Input Generator     #")
            print("##################################")
            print("\n### Status ###")
            print(f" Eq: [{status_eq}]")

            # TODO: separate by 'choices'
            print("\n### Main ###")

            self._read_equilibrium()
            self._get_midplane_mapping()

            self._read_profiles()
            self._interpolate_profiles() 
            self.plot_profiles()

            self._determine_resolutions()
            self.plot_resolutions()

            self._generate_surfaces()
            self.plot_surfaces()

            self._write_tomms_input()

            print("----------------------------------")

            # exiting
            print(">> Done")
            break

if __name__=='__main__':
    tomms_ingen = TommsInputGenerator()
