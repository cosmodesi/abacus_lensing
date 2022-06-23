# TODO: drj I think you need to think about whether files record first or last comoving
# could break into 0.1 to 0.5, 0.5 to 0.8, 0.8 to 1.1, 1.1 to 1.4, 1.4 to 1.6, 1.6 to 2.5

import os
import gc
import sys
import glob
import time

import numpy as np
import healpy as hp
import asdf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from numba import njit, prange, set_num_threads
import ducc0

from fast_cksum.cksum_io import CksumWriter
from util import histogram_hp, add_kappa_shell, add_shell
from tools import extract_steps, save_asdf, compress_asdf
from mask_kappa import get_mask
from abacusnbody.metadata import get_meta

# each array is 12 GB for 16384

@njit(parallel=True)
def fast_ring2nest(hp_map, hp_ring2nest, hp_mask):
    new_map = np.zeros(len(hp_map), dtype=hp_map.dtype)
    for i in prange(len(hp_map)):
        #if masking: # if None not happy
        new_map[i] = hp_map[hp_ring2nest[i]] * hp_mask[i]
        #else:
        #new_map[i] = hp_map[hp_ring2nest[i]]
    return new_map

# adding CMB a bit ad hoc
z_cmb = 1089.276682
chi_cmb = 13872.661199427605 # Mpc

# select the source redshifts (0.1 to 2.5 delta z = 0.05, around 50)
#z_srcs = np.arange(0.1, 2.5, 0.05);
z_dic = {}
e_tol = 1.e-6
z_srcs = np.arange(0.15, 0.45-e_tol, 0.05); z_str = "_z0.15_z0.45"; z_dic[z_str] = z_srcs
z_srcs = np.arange(0.45, 0.75-e_tol, 0.05); z_str = "_z0.45_z0.75"; z_dic[z_str] = z_srcs
z_srcs = np.arange(0.75, 1.05-e_tol, 0.05); z_str = "_z0.75_z1.05"; z_dic[z_str] = z_srcs
z_srcs = np.arange(1.05, 1.35-e_tol, 0.05); z_str = "_z1.05_z1.35"; z_dic[z_str] = z_srcs
z_srcs = np.arange(1.35, 1.65-e_tol, 0.05); z_str = "_z1.35_z1.65"; z_dic[z_str] = z_srcs
z_srcs = np.arange(1.65, 1.95-e_tol, 0.05); z_str = "_z1.65_z1.95"; z_dic[z_str] = z_srcs
z_srcs = np.arange(1.95, 2.25-e_tol, 0.05); z_str = "_z1.95_z2.25"; z_dic[z_str] = z_srcs
z_srcs = np.arange(2.25, 2.55-e_tol, 0.05); z_srcs[-1] = z_cmb; z_str = "_z2.2_z2.45_cmb"; z_dic[z_str] = z_srcs

# healpix parameters (true for all AbacusSummit products, but could automize)
nside = 16384
delta_omega = hp.nside2pixarea(nside) # steradians per pixel
npix = (hp.nside2npix(nside))

# number of threads
nthreads = 16

# simulation name
#sim_name = f"AbacusSummit_base_c000_ph{i:03d}"
sim_name = sys.argv[1] #"AbacusSummit_base_c000_ph000"

# directories
header_dir = f"/global/homes/b/boryanah//repos/abacus_lc_cat/data_headers/{sim_name}/"
heal_dir = f"/global/project/projectdirs/desi/cosmosim/Abacus/{sim_name}/lightcones/heal/"
save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"
os.makedirs(save_dir, exist_ok=True)

# all healpix file names
hp_fns = sorted(glob.glob(heal_dir+"LightCone*.asdf"))
n = len(hp_fns)

# simulation parameters
header = asdf.open(hp_fns[0])['header']
Lbox = header['BoxSize'] # 2000. # Mpc/h
PPD = header['ppd'] # 6912
NP = PPD**3

# cosmological parameters
Om_m = header['Omega_M'] #0.315192
H0 = header['H0']
h = H0/100.  # 0.6736
c = 299792.458 # km/s

# comoving particle density in 1/Mpc^3
n_part = NP/(Lbox/h)**3

# all snapshots and redshifts that have light cones; early to recent redshifts
zs_all = np.load(header_dir+"redshifts.npy")
#zs_all[-1] = float("%.1f" % zs_all[-1])

# ordered from small to large; small step number to large step number
steps_all = np.load(header_dir+"steps.npy")

# comoving distances in Mpc/h; far shells to close shells
chis_all = np.load(header_dir+"coord_dist.npy")
chis_all /= h # Mpc

# get functions relating chi and z
z_min = 0.1
chi_min = get_meta(sim_name, redshift=z_min)['CoordinateDistanceHMpc']/h
z_edges = np.append(zs_all, np.array([z_min]))
chi_edges = np.append(chis_all, np.array([chi_min]))
chi_of_z = interp1d(z_edges, chi_edges)
z_of_chi = interp1d(chi_edges, z_edges)
z_mid = (z_edges[1:]+z_edges[:-1])*.5
chi_mid = chi_of_z(z_mid)

# furthest and closest shells are at what time step
step_min = np.min(steps_all)
step_max = np.max(steps_all)

# location of the observer
origin = np.array([-990., -990., -990.])

# distance from furthest point to observer in Mpc/h
chi_max = 1.5*Lbox-origin[0]
chi_max /= h # Mpc

# select the final and initial step for computing the convergence map
step_start = steps_all[np.argmax((chis_all-chi_max) < 0)] # corresponds to 4000-origin
step_stop = step_max
z_max = zs_all[np.argmax((chis_all-chi_max) < 0)]
print("starting step = ", step_start)
print("furthest redshift = ", z_max)

# for the dowgrading
new_nside = 16384 # to avoid overflow for healpy
new_npix = (hp.nside2npix(new_nside))

# starting redshift
z_start = 0.1
z_stop = z_cmb

# ring2nest preload
t = time.time()
new_ipix = np.arange(new_npix)
#new_ring2nest = hp.ring2nest(new_nside, new_ipix)
base = ducc0.healpix.Healpix_Base(nside, "RING")
new_ring2nest = base.ring2nest(new_ipix, nthreads=nthreads)
print("time preload ring2nest = ", time.time()-t)
del new_ipix
gc.collect()

# maximum chi where the mask is the same
chi_same_mask = 1980./h
print("z where same mask = ", z_of_chi(chi_same_mask))

sum = 0

want_mask = True
if want_mask and z_start <= z_of_chi(chi_same_mask):
    t = time.time()
    # get mask at this redshift source
    try:
        mask = asdf.open(save_dir+f"mask_{sum:05d}.asdf", lazy_load=True, copy_arrays=True)['data']['mask']
        print("worked!")
        quit()
    except:
        mask = get_mask(new_nside, chi_same_mask*h, sim_name).astype(np.float32)
    print("time getting mask = ", time.time()-t)
else:
    mask = None
    
for z_str in z_dic.keys():

    # redshift and comoving distance to the lensing sources in Mpc/h
    z_srcs = z_dic[z_str]
    
    for i in range(len(z_srcs)):

        if z_srcs[i] < z_start: sum += 1; continue
        if z_srcs[i] > z_stop: sum += 1; quit()
        
        if want_mask and z_srcs[i] > z_of_chi(chi_same_mask):
            t = time.time()
            try:
                del mask; gc.collect()
            except:
                pass
            # if CMB, must be at chi_max
            if z_srcs[-1] > z_max:
                try:
                    mask = asdf.open(save_dir+f"mask_{sum:05d}.asdf", lazy_load=True, copy_arrays=True)['data']['mask']
                except:
                    mask = get_mask(new_nside, chis_all[np.argmax((chis_all-chi_max) < 0)]*h, sim_name).astype(np.float32)
            else:
                try:
                    mask = asdf.open(save_dir+f"mask_{sum:05d}.asdf", lazy_load=True, copy_arrays=True)['data']['mask']
                except:
                    mask = get_mask(new_nside, chi_of_z(z_srcs[i])*h, sim_name).astype(np.float32)
            print("time getting mask = ", time.time()-t)
        elif not want_mask:
            mask = None

        
        # load kappas
        t = time.time()
        kappas = asdf.open(save_dir+f"kappas{z_str}.asdf", lazy_load=True, copy_arrays=True)['data']['Kappas']
        #kappas = np.zeros((len(z_srcs), npix), dtype=np.float32)
        
        # dictionary containing info about kappas
        header = {}
        header['SourceRedshift'] = z_srcs[i]
        header['SimulationName'] = f'{sim_name}'
        header['HEALPix_nside'] = new_nside
        header['HEALPix_order'] = 'RING'
        print("header = ", header.items())
        
        # to save space
        kappa = kappas[i, :]
        del kappas
        gc.collect()
        print("time loading = ", time.time()-t)
        assert kappa.dtype == np.float32

        # downgrade to new nside
        if new_nside != nside:
            t = time.time()
            kappa = hp.ud_grade(kappa, new_nside, order_in='NESTED')
            print("time downgrade = ", time.time()-t)
        
        # convert to ring (checked)
        t = time.time()
        table = {}
        table['kappa'] = fast_ring2nest(kappa, new_ring2nest, mask) #equiv to kappa[new_ring2nest]
        print("time ring2nest = ", time.time()-t)
        
        # string names by which to identify masks (column names and header)
        kappa_fn = f"kappa_{sum:05d}"
        compress_asdf(save_dir+f"{kappa_fn}.asdf", table, header)
        
        sum += 1
        del kappa, table
        gc.collect()
quit()



# string names by which to identify masks (column names and header)
gamma1_strs = [f"Gamma1_{i:05d}" for i in range(len(z_srcs))]
gamma2_strs = [f"Gamma2_{i:05d}" for i in range(len(z_srcs))]

# dictionary containing info about gammas
table1 = {}
header1 = {}
header1['SourceRedshifts'] = z_srcs
header1['ColumnNames'] = gamma1_strs+gamma2_strs
header1['SimulationName'] = f'{sim_name}'
header1['HEALPix_nside'] = new_nside
header1['HEALPix_order'] = 'RING'
print("header1 = ", header1.items())

# string names by which to identify masks (column names and header)
alpha1_strs = [f"Alpha1_{i:05d}" for i in range(len(z_srcs))]
alpha2_strs = [f"Alpha2_{i:05d}" for i in range(len(z_srcs))]

# dictionary containing info about gammas
table2 = {}
header2 = {}
header2['SourceRedshifts'] = z_srcs
header2['ColumnNames'] = alpha1_strs + alpha2_strs
header2['SimulationName'] = f'{sim_name}'
header2['HEALPix_nside'] = new_nside
header2['HEALPix_order'] = 'RING'
print("header2 = ", header2.items())

# mask smoothing scale
#smooth_scale_mask = (5./60.)*np.pi/180. # 5 arcmin
smooth_scale_mask = 0.

# make a loop over redshift sources
for i in range(len(z_srcs)):
    
    # source redshift and kappa
    z_src = z_srcs[i]
    kappa_src = table[kappa_strs[i]]
    print("kappa = ", kappa_src[:20])

    # get mask at this redshift source
    mask_src = get_mask(new_nside, chi_of_z(z_src)*h, sim_name).astype(np.float32)
    print("mask = ", mask_src[:20])
    
    # fraction unmasked
    fsky = np.sum(mask_src**2)/len(mask_src)
    print("fsky before smoothing = ", fsky)
    
    # apply apodization
    if smooth_scale_mask > 0.:
        mask_src = hp.smoothing(mask_src, fwhm=smooth_scale_mask, iter=5, verbose=False)     
    fsky = np.sum(mask_src**2)/len(mask_src)
    print("fsky after smoothing = ", fsky)

    # mask kappa
    kappa_masked = kappa_src * mask_src
    print("kappa masked = ", kappa_masked[:20])
    
    # Convert convergence map to alms via spherical harmonics.
    kelm = hp.map2alm(kappa_masked)

    # Get the lmax and l,m values of the alms.
    lmax = hp.Alm.getlmax(len(kelm))
    l, m = hp.Alm.getlm(lmax)

    # Create the zero-initialized shear alms.
    gtlm = np.zeros_like(kelm)
    gelm = np.zeros_like(kelm)
    gblm = np.zeros_like(kelm)

    # Compute the E mode shear map alms and set to zero for l = 0 to avoid division by zero.
    good_ls = l > 0
    l = l[good_ls]
    gelm[good_ls] = - np.sqrt((l + 2) * (l - 1) / (l * (l + 1))) * kelm[good_ls]

    # get potential, psi
    gtlm[good_ls] = -2. / (l * (l + 1)) * kelm[good_ls]
    print("filtering for deflection and gamma begins")
    
    # take derivative of psi (alpha = grad psi = alpha_theta e_theta + alpha_phi e_phi, so alpha_theta = dpsi/dtheta, alpha_phi = dpsi/stheta/dphi) 
    psi, dpsidth, dpsisthdph = hp.alm2map_der1(gtlm, new_nside)
    table2[alpha1_strs[i]] = dpsidth
    table2[alpha2_strs[i]] = dpsisthdph
    print("alpha_theta, alpha_phi = ", dpsidth[:20], dpsisthdph[:20])
    
    # Get the real-space shear map components using spin-2 spherical harmonics.
    _, gamma1, gamma2 = hp.alm2map([gtlm, gelm, gblm], new_nside, lmax=lmax, verbose=False)
    table1[gamma1_strs[i]] = gamma1
    table1[gamma2_strs[i]] = gamma2
    print("gamma1,2 = ", gamma1[:20], gamma2[:20])
    
f.close()

# compress table and save into asdf file
compress_asdf(save_dir+"gammas.asdf", table1, header1)
compress_asdf(save_dir+"alphas.asdf", table2, header2)

# for measuring power spectrum (kappa)
#cl_kszsq_gal = hp.anafast(cmb_fltr_masked**2, gal_masked, lmax=LMAX-1, pol=False)/fsky

# power parameters
LMIN = 0
LMAX = 3*nside+1
ell_data = np.arange(LMIN, LMAX, 1)
