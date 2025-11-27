import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg') # to enable interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from scipy.interpolate import interp1d, interp2d, RectBivariateSpline # RectBivariateSpline is often better than interp2d
from pathlib import Path

import configparser

PROTON_MASS   = 1.6720e-27  # kg
ELECTRON_MASS = 9.1094e-31  # kg
UNIT_CHARGE   = 1.6022e-19  # C

PREFIX_ERRORS = ' * '

# some general helper functions (TODO: make a separate module)
def is_monotonic(arr):
  diffs = np.diff(arr)
  return np.all(diffs >= 0) or np.all(diffs <= 0)

def densify_line(r_coords, z_coords, max_seg_length):
    new_r = [r_coords[0]]
    new_z = [z_coords[0]]
    
    # Iterate over each segment (from point i to i+1)
    for i in range(len(r_coords) - 1):
        r1, z1 = r_coords[i], z_coords[i]
        r2, z2 = r_coords[i+1], z_coords[i+1]
        
        # Calculate the length of this segment
        dist = np.sqrt((r2 - r1)**2 + (z2 - z1)**2)
        
        if dist > max_seg_length:
            # If segment is too long, subdivide it
            num_segments = int(np.ceil(dist / max_seg_length))
            
            # Generate new points using linspace
            # We skip the first point [0] because it's r1/z1 (already added)
            r_interp = np.linspace(r1, r2, num_segments + 1)[1:]
            z_interp = np.linspace(z1, z2, num_segments + 1)[1:]
            
            new_r.extend(r_interp)
            new_z.extend(z_interp)
        else:
            # Segment is short enough, just add the end point
            new_r.append(r2)
            new_z.append(z2)
            
    return np.array(new_r), np.array(new_z)

class Eqdsk:
    def __init__(self, filename):
        self.filename = filename
        print(f">> EQDSK")
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
                raise EOFError(f"{PREFIX_ERRORS}Expected {num} values, but reached end of file after reading {items_read}.")
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
                        raise ValueError(f"{PREFIX_ERRORS}Could not convert '{line[start_index:end_index]}' to float. Line: '{line}'")
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
            self.header = line_str[:line_str.rfind(line_parts[-3])].strip() # Get text header part
            self.idum, self.nw, self.nh = map(int, line_parts[-3:])

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
            self.pp    = self._read_1d(f, self.nw) # P'. Should add (-) sign in front?
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
            print(f"{PREFIX_ERRORS}Warning: No rzsep. Equilibrium data might not be fully processed.")
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
            print(f"{PREFIX_ERRORS}Warning: Calculated area is close to zero. Volume calculation skipped.")
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
        cntr = ax.contour(self.r, self.z, self.psinrz.T, levels=np.linspace(0, 1.2, 101)) # Transpose psinrz
        fig.colorbar(cntr, ax=ax, label=r'Normalized Poloidal Flux ($\psi_N$)')
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
        print("##################################")
        print("#      TOMMS Input Generator     #")
        print("##################################")

        # setups
        self.params = self._default_parameters()
        self._load_parameters(filepath='./params.in')
        self.print_parameters()

        self.eq          = None
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
        self.files_written         = {'surf': False, 
                                      'dpol': False,
                                      'wall': False}

        # interactive loop
        self._run_interface()

    def _default_parameters(self):
        params = {}

        params['g_file']  = './inputs/g184833.04800_new'
        params['te_file'] = './inputs/te_d184833_4800_pfile_new.prf'
        params['ti_file'] = './inputs/ti_d184833_4800_pfile_adj_08.prf'
        params['ne_file'] = './inputs/ne_d184833_4800_pfile_new.prf'

        # if external wall input
        params['wall_file'] = ''

        # general settings
        params['num_mid']        = 1000
        params['dr_scale_psin']  = [0.0, 0.2, 0.3, 0.6, 0.8, 1.0, 1.05, 1.07, 1.1, 1.2]
        params['dr_scale_fac']   = [2.0, 2.0, 1.2, 1.0, 1.0, 1.0, 1.2,  1.5,  4.0, 12.0]
        params['dr_min']         = 1e-3 # m
        params['pol_scale_fac']  = 1.3
        params['pol_core_reduction_fac'] = 1.0 # 0.3

        # outputs
        params['output_dir']       = './outputs'
        params['output_surf_file'] = 'surf_xgc1.txt'
        params['output_dpol_file'] = 'dpol_xgc1.txt'
        params['output_wall_file'] = 'wallcurve.txt'
    
        return params

    def _load_parameters(self, filepath='./params.in'):
        print(f"\n>> loading parameters from {filepath}")
        if not os.path.exists(filepath):
            print(f"{PREFIX_ERRORS}Warning: Config file '{filepath}' not found. Using defaults.")
            return

        config = configparser.ConfigParser()
        try:
            config.read(filepath)
        except configparser.Error as e:
            print(f"{PREFIX_ERRORS}Error parsing config file '{filepath}': {e}")
            return

        default_keys   = set(self.params.keys())
        overriden_keys = set()
        
        for section in config.sections():
            for key in config[section]:
                # 1. Check if key is valid (matches a default)
                if key not in self.params:
                    print(f"{PREFIX_ERRORS}Warning: Unknown key '{key}' in config. Ignoring.")
                    continue  # Skip to the next key

                # 2. Get the string value from config
                value_str = config[section][key]
                
                # 3. Get the type from the default parameter
                default_val = self.params[key]
                default_type = type(default_val)

                try:
                    # 4. Parse the string based on the default's type
                    if isinstance(default_val, bool):
                        # Use configparser's built-in boolean handler
                        parsed_value = config[section].getboolean(key)
                    
                    elif isinstance(default_val, list):
                        # Get element type from default list (if list isn't empty)
                        el_type = type(default_val[0]) if default_val else str
                        # Split by comma, strip whitespace, and cast each element
                        parsed_value = [el_type(v.strip()) for v in value_str.split(',')]
                    
                    elif isinstance(default_val, (int, float, str)):
                        # Simple direct cast for int, float, or str
                        parsed_value = default_type(value_str)
                    
                    else:
                        # Fallback for other types (less common)
                        parsed_value = default_type(value_str)

                    # 5. Success: Override the parameter
                    self.params[key] = parsed_value

                    overriden_keys.add(key)

                except Exception as e:
                    # Handle errors (e.g., casting 'hello' to int)
                    print(f"{PREFIX_ERRORS}Error parsing key '{key}' with value '{value_str}'. "
                          f"{PREFIX_ERRORS}Expected type {default_type}. Error: {e}")

        keys_left_as_default = default_keys - overriden_keys
        if keys_left_as_default:
            print("... following parameters were left as default:")
            for key in self.params.keys():
                if key in keys_left_as_default:
                    print(f"  - {key}")
        else:
            print("... all default parameters were set by the config file.")

    def print_parameters(self):
        print("-------  Parameters -------")
        # Find the longest key for alignment
        max_len = max(len(key) for key in self.params) + 2 # Add 2 for padding
        
        # Loop through keys and print
        for key in self.params.keys():
            value = self.params[key]
            
            value_str = ""

            # to format lists aligned
            if (isinstance(value, list) and 
                value and 
                all(isinstance(x, (int, float)) for x in value)):
                
                formatted_list = [f"{x:5.2f}" for x in value]
                
                # Join them back into a string
                value_str = "[" + ", ".join(formatted_list) + "]"
            
            else:
                # For everything else (strings, single ints, etc.), just use str()
                value_str = str(value)

            print(f"  {key:<{max_len}} = {value_str}")
            
        print("---------------------------\n")

    def _read_equilibrium(self):
        try: 
            self.eq = Eqdsk(self.params['g_file'])
            self.equilibrium_loaded = True
        except Exception as e:
            print(f"{PREFIX_ERRORS}Error loading equilibrium file: {e}")
            self.equilibrium_loaded = False

        self.eq.plot_overview()

    def _get_midplane_mapping(self):
        print("\n>> get midplane mapping")
        if not self.equilibrium_loaded:
            print(f"{PREFIX_ERRORS}Warning: Load equilibrium first.")
            return

        if self.eq.rzlim.size == 0:
            print(f"{PREFIX_ERRORS}Warning: No limiter information in the Eqdsk")

            if self.params['wall_file']:
                print(f"{PREFIX_ERRORS}Warning: Reading external wall file {self.params['wall_file']}")
                try:
                    self.eq.rzlim = np.stack(self._read_prf(self.params['wall_file'], 'wall'), axis=1)
                except ValueError:
                    raise ValueError(f"...failed to read wall file {self.params['wall_file']}")

            else:
                raise ValueError(f"{PREFIX_ERRORS}Error: no wall information neither from EQDSK nor externally given")

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
            print(f"{PREFIX_ERRORS}Warning: midplane['psin'] is not monotonic")

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
                    raise EOFError(f"{PREFIX_ERRORS}Expected {n} data lines, but file ended after {lines_read} lines.")
                try:
                    psi[l], var[l] = map(float, line.strip().split())
                    lines_read += 1
                except ValueError:
                    raise ValueError(f"{PREFIX_ERRORS}Could not parse line {l+2} as two floats: '{line.strip()}'")

            # --- Optionally check end flag ---
            end_flag_line = file.readline()
            if end_flag_line: # Check if a line was actually read
                try:
                    end_flag = int(end_flag_line.strip().split()[0])
                    if end_flag != -1:
                        print(f"{PREFIX_ERRORS}Warning: Expected end flag -1 in {filename}, but found {end_flag}.\n...Proceeding anyway.")
                except (ValueError, IndexError):
                    print(f"{PREFIX_ERRORS}Warning: Could not parse end flag line in {filename}: '{end_flag_line.strip()}'.\n...Proceeding anyway.")
            else:
                # No end flag line found, which is okay based on your requirement
                print(f"{PREFIX_ERRORS}Warning: No end flag line found in {filename} after reading {n} data points.\n...Proceeding anyway.")

            return psi, var

    def _read_profiles(self):
        print("\n>> read input profiles")
        if not self.equilibrium_loaded:
            print(f"{PREFIX_ERRORS}Warning: Load equilibrium first.")
            return
        try:
            print("\n>> load te, ti, ne profiles")

            psi_te, te = self._read_prf(self.params['te_file'], 'Te')
            psi_ti, ti = self._read_prf(self.params['ti_file'], 'Ti')
            psi_ne, ne = self._read_prf(self.params['ne_file'], 'ne')
            self.prof = {'psi_te': psi_te, 'te': te,
                         'psi_ti': psi_ti, 'ti': ti,
                         'psi_ne': psi_ne, 'ne': ne}
            self.profiles_loaded = True
        except Exception as e:
            print(f"{PREFIX_ERRORS}Error loading profile files: {e}")
            self.profiles_loaded = False        

    def _interpolate_profiles(self):
        print("\n>> interpolate profiles onto midplane grid")
        if not self.midplane_setted or not self.profiles_loaded:
            print(f"{PREFIX_ERRORS}Warning: Midplane mapping and raw profiles must be loaded first.")
            return

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
        self.prof_interp['te'][self.prof_interp['te'] < 0] = 1e-3
        self.prof_interp['ti'][self.prof_interp['ti'] < 0] = 1e-3
        self.prof_interp['ne'][self.prof_interp['ne'] < 0] = 1e-3

        self.profiles_interpolated = True

    def plot_profiles(self):
        if not self.profiles_loaded:
            print(f"{PREFIX_ERRORS}Warning: Profiles to plot have not been loaded yet")
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

        ax.set_xlabel(r'Normalized Poloidal Flux ($\psi_N$)')
        ax.set_ylabel('a.u.')
        ax.set_title('Initial Raw Profiles')
        ax.grid(True, alpha=0.5)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def _inspect_profiles(self):
        print("\n>> inspect resolution")
        if not self.profiles_interpolated:
            print(f"{PREFIX_ERRORS}Warning: Interpolated profiles needed")
            return

    def _determine_resolutions(self):
        print("\n>> determine resolution")
        if not self.profiles_interpolated:
            print(f"{PREFIX_ERRORS}Warning: Interpolated profiles needed")
            return

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
        rhoi = np.sqrt(mi*ti_ev*UNIT_CHARGE)/(UNIT_CHARGE*Bmid) 

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
            print(f"{PREFIX_ERRORS}Warning: Resolutions have not been determined yet")
            return

        rmid        = self.midplane['r']
        psin        = self.midplane['psin']
        rhoi        = self.resolution['rhoi']
        dr_target   = self.resolution['dr_target']
        dpol_target = self.resolution['dpol_target']

        fig,ax = plt.subplots(figsize=(6,5))

        ax.plot(psin, rhoi, c='tab:blue', label='rhoi')
        ax.plot(psin, dr_target, c='tab:orange', label='dr_target')
        ax.plot(psin, dpol_target, c='tab:green', label='dpol_target')

        ax.set_xlabel(r'Normalized Poloidal Flux ($\psi_N$)')
        ax.set_ylabel('Length Scale [m]')
        ax.set_title('Target Resolution vs. Ion Gyroradius')
        ax.legend()
        ax.grid(True, alpha=0.5)

        plt.tight_layout()
        plt.show()

    def _generate_surfaces(self):
        print("\n>> generate surfaces")
        if not self.resolution_determined:
            print(f"{PREFIX_ERRORS}Warning: Resolution have not been determined yet")
            return

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
            print(f"{PREFIX_ERRORS}Warning: Surfaces have not been determined yet")
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

    def _edit_wall_interactive(self):
        print("\n>> edit wall")
        if not self.equilibrium_loaded:
            print(f"{PREFIX_ERRORS}Warning: Load equilibrium first.")
            return

        print("--- Interactive Wall Editor ---")
        print(" - Click near a point on the 'Original Limiter' (black line) to add/remove it.")
        print(" - Added points form the 'Simplified Wall' (red line).")
        print(" - Points are added/removed maintaining original order.")
        print(" - Close the plot window when finished.")
        print("-------------------------------\n")

        # Define a desired resolution.
        # intially 1% of the total R-width of the plot
        f_densify = 0.01
        max_seg_length = f_densify * (np.max(self.eq.rzlim[:,0]) - np.min(self.eq.rzlim[:,0]))
        original_limiter_r, original_limiter_z = densify_line(self.eq.rzlim[:,0], self.eq.rzlim[:,1] ,max_seg_length)

        # --- Setup Plot ---
        fig, ax = plt.subplots(figsize=(8, 10))

        # space for a textbox to draw a target psin contour
        fig.subplots_adjust(bottom=0.15)

        # Plot psi contour for guidance
        cntr = ax.contour(self.eq.r, self.eq.z, self.eq.psinrz.T, levels=np.linspace(0, 1.2, 101)) # Transpose psinrz
        fig.colorbar(cntr, ax=ax, label=r'Normalized Poloidal Flux ($\psi_N$)')
        ax.plot(self.eq.rzsep[:, 0], self.eq.rzsep[:, 1], 'tab:orange', linewidth=2, label='Separatrix')
        ax.plot(self.eq.rmag, self.eq.zmag, 'kx', markersize=10, mew=2, label='Magnetic Axis')

        # Plot original limiter - store handle for hover effects if desired later
        densified_scatter = ax.scatter(original_limiter_r, original_limiter_z, c='gray', label=f'densified ({f_densify*100:.1f}%)', s=8)
        ax.plot(self.eq.rzlim[:,0], self.eq.rzlim[:,1], 'k.-', label='Original Limiter', markersize=4)

        # Plot currently selected points (initially empty) - store line handle
        selected_line, = ax.plot([], [], 'ro-', label='Simplified Wall (Click to add)', markersize=6)

        ax.set_xlabel('R [m]')
        ax.set_ylabel('Z [m]')
        ax.set_title('Interactive Wall Selection\n(Close window when done)')
        ax.axis('equal')
        ax.legend()
        ax.grid(True, alpha=0.5)

        # --- Interactive ---
        self.wall              = {} # Reset selected points for this session
        selected_indices       = set() # Keep track of indices added
        target_contour_storage = [None]

        def submit_psi_target(text):
            try:
                # 1. Get the value from the text box
                psin_val = float(text)
            except ValueError:
                print(f"{PREFIX_ERRORS}Invalid value. Please enter a number (e.g., '1.05').")
                return

            # 2. Remove the *old* contour set, if one exists
            if target_contour_storage[0]:
                try:
                    # New Matplotlib (3.8+) - remove the whole object directly
                    target_contour_storage[0].remove()
                except AttributeError:
                    # Old Matplotlib (Pre-3.8) - remove individual collections
                    for collection in target_contour_storage[0].collections:
                        collection.remove()
                target_contour_storage[0] = None

            # 3. Draw the new contour
            print(f"... Drawing target contour at psi_N = {psin_val}")
            try:
                target_cs = ax.contour(
                    self.eq.r, self.eq.z, self.eq.psinrz.T, 
                    levels=[psin_val], 
                    colors=['cyan'],  # Use a bright color
                    linestyles=['--'],
                    linewidths=[2.0]
                )
                
                # 4. Store the new contour set
                target_contour_storage[0] = target_cs
                fig.canvas.draw_idle() # Redraw the canvas
            except Exception as e:
                print(f"{PREFIX_ERRORS}Could not draw contour: {e}")

        def submit_f_densify(text):
            # Use 'nonlocal' to modify the variables from the outer scope
            nonlocal f_densify, original_limiter_r, original_limiter_z
            
            try:
                new_val = float(text)
                if not (0 < new_val <= 1.0):
                    raise ValueError("must be > 0 and <= 1.0")
            except Exception as e:
                print(f"{PREFIX_ERRORS}Invalid f_densify value: {e}")
                return

            f_densify = new_val
            print(f"... Re-densifying wall with f_densify = {f_densify}")

            # 1. Recalculate the densified line
            max_seg_length = f_densify * (np.max(self.eq.rzlim[:,0]) - np.min(self.eq.rzlim[:,0]))
            original_limiter_r, original_limiter_z = densify_line(self.eq.rzlim[:,0], self.eq.rzlim[:,1] ,max_seg_length)

            # 2. Update the scatter plot data
            new_offsets = np.column_stack((original_limiter_r, original_limiter_z))
            densified_scatter.set_offsets(new_offsets)
            
            # 3. Update the label and redraw the legend
            densified_scatter.set_label(f'densified ({f_densify*100:.1f}%)')
            ax.legend()
            
            # 4. CRITICAL: Reset the user's selection
            selected_indices.clear()
            self.wall.clear()
            selected_line.set_data([], [])
            print("... Selection has been reset due to re-densification.")

            # 5. Redraw the canvas
            fig.canvas.draw_idle()

        # Create the axes for the textboxes [left, bottom, width, height]
        ax_box_densify = fig.add_axes([0.15, 0.05, 0.3, 0.05])
        text_box_densify = TextBox(ax_box_densify, 'Densify (0-1):', initial=str(f_densify))
        text_box_densify.on_submit(submit_f_densify)

        ax_box_psi = fig.add_axes([0.6, 0.05, 0.3, 0.05]) #([0.3, 0.05, 0.4, 0.05])
        text_box_psi = TextBox(ax_box_psi, r'Target $\psi_N$:', initial='1.05')
        text_box_psi.on_submit(submit_psi_target)

        def onclick(event):
            if event.inaxes != ax: return # Ignore clicks outside the plot area
            if event.button != 1: return # Ignore right/middle clicks

            click_r, click_z = event.xdata, event.ydata
            if click_r is None or click_z is None: return # Ignore clicks outside data range

            # Calculate distance from click to all original limiter points
            distances = np.sqrt((original_limiter_r - click_r)**2 + (original_limiter_z - click_z)**2)

            # Find the index of the closest original point
            idx_min = np.argmin(distances)
            min_dist = distances[idx_min]

            # Define a tolerance margin (adjust as needed, depends on plot scale)
            r_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            margin = 0.05 * r_range # 5% margin

            if min_dist < margin:
                # add clicked points
                if idx_min not in selected_indices: # add clicked point
                    print(f"Adding point {idx_min}: (R={original_limiter_r[idx_min]:.3f}, Z={original_limiter_z[idx_min]:.3f})")
                    selected_indices.add(idx_min)
                else: # remove if already selected
                    print(f"Removing point {idx_min}")
                    selected_indices.remove(idx_min)

                # Rebuild the simplified wall list, maintaining original order
                if len(selected_indices)>0:
                    sorted_indices = sorted(list(selected_indices))
                    self.wall['r'] = original_limiter_r[sorted_indices]
                    self.wall['z'] = original_limiter_z[sorted_indices]
                else:
                    self.wall = {} # empty if no indices selected

                # Update the plot data for the red line
                if len(self.wall['r']) > 0:
                    selected_line.set_data(self.wall['r'], self.wall['z'])
                else:
                    selected_line.set_data([], [])

                # Redraw the canvas
                fig.canvas.draw_idle()
            else:
                print("Click is too far from any point. Try clicking closer.")

        # Connect the click event handler
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # Show the plot - execution pauses here until the window is closed
        plt.show(block=True)

        # Disconnect the event handler after the plot is closed
        fig.canvas.mpl_disconnect(cid)

        # Convert list of points to a NumPy array after closing
        if self.wall:
            self.wall['r'] = np.array(self.wall['r'])
            self.wall['z'] = np.array(self.wall['z'])

            # Append the first point to the end if it's not already the same
            first_point = [self.wall['r'][0], self.wall['z'][0]]
            last_point  = [self.wall['r'][-1], self.wall['z'][-1]]
            if not np.allclose(first_point, last_point):
                 self.wall['r'] = np.append(self.wall['r'], self.wall['r'][0])
                 self.wall['z'] = np.append(self.wall['z'], self.wall['z'][0])
                 print("... Loop closed by appending the first point.")

            self.wall_curve_generated = True
            print(f"\nFinished selection. Simplified wall has {len(self.wall['r'])} points (loop closed).")

            # Plot the final closed loop
            fig_final, ax_final = plt.subplots(figsize=(8, 10))

            cntr = ax_final.contour(self.eq.r, self.eq.z, self.eq.psinrz.T, levels=np.linspace(0, 1.2, 61)) # Transpose psinrz
            fig.colorbar(cntr, ax=ax_final, label=r'Normalized Poloidal Flux ($\psi_N$)')
            ax_final.plot(self.eq.rzsep[:, 0], self.eq.rzsep[:, 1], 'tab:orange', linewidth=2, label='Separatrix')
            ax_final.plot(self.eq.rmag, self.eq.zmag, 'kx', markersize=10, mew=2, label='Magnetic Axis')

            ax_final.plot(original_limiter_r, original_limiter_z, 'k.-', label='Original Limiter', markersize=4, alpha=0.3)
            ax_final.plot(self.wall['r'], self.wall['z'], 'ro-', label='Final Simplified Wall (Closed)', markersize=6)
            ax_final.scatter(first_point[0], first_point[1], facecolors='none', edgecolors='b')
            ax_final.set_xlabel('R [m]')
            ax_final.set_ylabel('Z [m]')
            ax_final.set_title('Final Selected Wall Curve')
            ax_final.axis('equal')
            ax_final.legend()
            ax_final.grid(True, alpha=0.5)
            plt.show()

            self.wall_generated = True
        else:
            self.wall_generated = False
            print("\nFinished selection. No points selected for simplified wall.")

    def _write_tomms_input(self):
        print("\n>> write TOMMS input")
        if not self.surface_generated:
            print(f"{PREFIX_ERRORS}Warning: Surfaces have not been determined yet")
            return

        if not self.wall_generated:
            print(f"{PREFIX_ERRORS}Warning: no simplified wall curve has been generated. Using rzlim in Eqdsk.")
            try:
                self.wall['r'] = self.eq.rzlim[:,0]
                self.wall['z'] = self.eq.rzlim[:,1]
                self.wall_generated = True
            except Exception as e:
                print(f"{PREFIX_ERRORS}Error: there is no limiter information in the Eqdsk {e}")
                self.wall_generated = False

        rsurf         = self.surface['r']
        nsurf         = len(rsurf)
        psurf_renorm  = self.surface['psin']
        pol_dist      = self.surface['dpol']

        output_dir = Path(self.params['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)

        surf_file  = output_dir / self.params['output_surf_file']
        dpol_file  = output_dir / self.params['output_dpol_file']
        wall_file  = output_dir / self.params['output_wall_file']

        # surface file
        try:
            with open(surf_file, 'w') as f:
                f.write(f"{nsurf}\n")
                for val in psurf_renorm:
                    f.write(f"{val:22.12e}\n")
            print(f"... {surf_file}")
            self.files_written['surf'] = True
        except Exception as e:
            print(f"{PREFIX_ERRORS}Error writing {surf_file}: {e}")
            return       

        # dpol file
        try:
            with open(dpol_file, 'w') as f:
                f.write(f"{nsurf}\n")
                for i in range(nsurf):
                    f.write(f"{psurf_renorm[i]:22.12e}   {pol_dist[i]:22.12e}\n")
            print(f"... {dpol_file}")
            self.files_written['dpol'] = True 
        except Exception as e:
            print(f"{PREFIX_ERRORS}Error writing {dpol_file}: {e}")

        # wall file
        if self.wall_generated:
            try:
                nwall = len(self.wall['r'])
                with open(wall_file, 'w') as f:
                    f.write(f"{nwall}\n")
                    for i in range(nwall):
                        f.write(f"{self.wall['r'][i]:22.12e}   {self.wall['z'][i]:22.12e}\n")
                print(f"... {wall_file}")
                self.files_written['wall'] = True 
            except Exception as e:
                print(f"{PREFIX_ERRORS}Error writing {wall_file}: {e}")

    def _run_interface(self):
        while True:
            # main
            self._read_equilibrium()
            self._get_midplane_mapping()

            self._read_profiles()
            self._interpolate_profiles() 
            self.plot_profiles()

            self._determine_resolutions()
            self.plot_resolutions()

            self._generate_surfaces()
            self.plot_surfaces()

            self._edit_wall_interactive()

            self._write_tomms_input()

            # exiting
            print("\n>> Done")
            break

if __name__=='__main__':
    tomms_ingen = TommsInputGenerator()
