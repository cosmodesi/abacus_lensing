# TODO: drj I think you need to think about whether files record first or last comoving
# could break into 0.1 to 0.5, 0.5 to 0.8, 0.8 to 1.1, 1.1 to 1.4, 1.4 to 1.6, 1.6 to 2.5

# can't reduce alms, can't go above 2*lmax; can do float32 and complex64 for the maps 

import os
import gc
import sys
import glob
import time

import numpy as np
import healpy as hp
import asdf
import ducc0
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from fast_cksum.cksum_io import CksumWriter
from util import histogram_hp, add_kappa_shell, add_shell
from tools import extract_steps, save_asdf, compress_asdf
from mask_kappa import get_mask
from abacusnbody.metadata import get_meta

# each array is 12 GB for 16384

# adding CMB a bit ad hoc
z_cmb = 1089.276682
chi_cmb = 13872.661199427605 # Mpc

# select the source redshifts (0.1 to 2.5 delta z = 0.05, around 50)
#z_srcs = np.arange(0.1, 2.5, 0.05);
z_dic = {}                      # 
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
new_nside = 16384 # to avoid overflow for healpy
lmax = 2*new_nside # twice 
base = ducc0.healpix.Healpix_Base(nside, "RING")
geom = base.sht_info()
delta_omega = hp.nside2pixarea(nside) # steradians per pixel
npix = (hp.nside2npix(nside))
new_npix = (hp.nside2npix(new_nside))

# Get the lmax and l,m values of the alms.
#lmax = hp.Alm.getlmax(len(kelm))
l, m = hp.Alm.getlm(lmax)
del m
gc.collect()

# Compute the E mode shear map alms and set to zero for l = 0 to avoid division by zero.
gamma_filter = -np.sqrt((l + 2) * (l - 1) / (l * (l + 1)))
phi_filter = -2. / (l * (l + 1))
gamma_filter[l == 0] = 0.
phi_filter[l == 0] = 0.
gamma_filter = gamma_filter.astype(np.float32)
phi_filter = phi_filter.astype(np.float32)
#gamma_filter = np.atleast_2d(gamma_filter)
#phi_filter = np.atleast_2d(phi_filter)
del l
gc.collect()

# number of threads
nthreads = 32

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

# smoothing scale
smooth_scale_mask = 0.
#smooth_scale_mask = (4./60.)*(np.pi/180.) # 4 arcmin

# maximum chi where the mask is the same
chi_same_mask = 1980./h
print("z where same mask = ", z_of_chi(chi_same_mask))

want_mask = False # we have already masked stuff
if want_mask:
    # get mask at this redshift source
    mask = get_mask(new_nside, chi_same_mask*h, sim_name).astype(np.float32)
    print("mask = ", mask[:20])

    # fraction unmasked
    fsky = np.sum(mask**2)/len(mask)
    print("fsky before smoothing = ", fsky)

    # apply apodization
    t = time.time()
    if smooth_scale_mask > 0.:
        mask = hp.smoothing(mask, fwhm=smooth_scale_mask, iter=3, verbose=False)     
    fsky = np.sum(mask**2)/len(mask)
    print("time apodizing = ", time.time()-t)
    print("fsky after smoothing = ", fsky)

sum = 0
for z_str in z_dic.keys():

    # redshift and comoving distance to the lensing sources in Mpc/h
    z_srcs = z_dic[z_str]
    
    for i in range(len(z_srcs)):

        # TESTING!!!!!!!!!!!!!!!!!!!!!!
        #if z_srcs[i] == 0.15: print("skipping 0.15"); sum += 1; continue
        
        # names of files
        kappa_fn = f"kappa_{sum:05d}.asdf"
        gamma_fn = f"gamma_{sum:05d}.asdf"
        alpha_fn = f"alpha_{sum:05d}.asdf"
        mask_fn = f"mask_{sum:05d}.asdf"
        
        # load kappa
        kappa = asdf.open(save_dir+kappa_fn, lazy_load=True, copy_arrays=True)['data']['kappa'][:]
        print("kappa should be float32 = ", kappa.dtype)

        # dictionary containing info about kappas
        header = {}
        header['SourceRedshift'] = z_srcs[i]
        header['SimulationName'] = f'{sim_name}'
        header['HEALPix_nside'] = new_nside
        header['HEALPix_order'] = 'RING'
        print("header = ", header.items())
        
        if want_mask:
            if z_srcs[i] > z_of_chi(chi_same_mask):
                # get mask at this redshift source
                # if CMB must be at chi_max
                if z_srcs[-1] > z_max:
                    mask = get_mask(new_nside, chis_all[np.argmax((chis_all-chi_max) < 0)]*h, sim_name).astype(np.float32)
                else:
                    mask = get_mask(new_nside, chi_of_z(z_srcs[i])*h, sim_name).astype(np.float32)
                print("mask = ", mask[:20])

                # fraction unmasked
                #fsky = np.sum(mask**2)/len(mask)
                #print("fsky before smoothing = ", fsky)
    
                # apply apodization
                if smooth_scale_mask > 0.:
                    mask = hp.smoothing(mask, fwhm=smooth_scale_mask, iter=3, verbose=False)     
                #fsky = np.sum(mask**2)/len(mask)
                #print("fsky after smoothing = ", fsky)
        
            # mask kappa
            kappa *= mask
            print("kappa masked = ", kappa.dtype, kappa[: 20])
        
        # turn into a vector
        kappa = np.atleast_2d(kappa)

        
        # Convert convergence map to alms via spherical harmonics.
        t = time.time()
        # if we wanted to apply ring weights to improve analysis accuracy, we wouldhave to do this here by hand
        kelm = ducc0.sht.experimental.adjoint_synthesis(lmax=lmax, spin=0, map=kappa, nthreads=nthreads, **geom)
        # normalize
        kelm *= 4*np.pi/(12*nside**2)
        print("should be complex 64 = ", kelm.dtype, kelm.shape)
        #kelm = hp.map2alm(kappa)
        print("time converting = ", time.time()-t)
        del kappa; gc.collect()
        
        # Create the zero-initialized shear alms.
        #gtlm = np.zeros_like(kelm)
        #gelm = np.zeros_like(kelm)
        #gblm = np.zeros_like(kelm)
        
        # get potential, psi
        #kelm = np.ones((1, len(phi_filter)), dtype=np.complex64)
        gtlm = (phi_filter * kelm.flatten())
        gtlm = np.atleast_2d(gtlm)
        print("filtering for deflection and gamma begins")

        
        # take derivative of psi (alpha = grad psi = alpha_theta e_theta + alpha_phi e_phi, so alpha_theta = dpsi/dtheta, alpha_phi = dpsi/stheta/dphi)
        t = time.time()
        #psi, dpsidth, dpsisthdph = hp.alm2map_der1(gtlm, new_nside, lmax=lmax)
        alpha = np.zeros((2, 12*nside**2), dtype=np.float32)
        ducc0.sht.experimental.synthesis_deriv1(alm=gtlm, map=alpha, lmax=lmax, nthreads=nthreads, **geom)
        print("time taking derivative = ", time.time()-t)

        # Save deflection field
        table = {}
        # load mask here and multiply and then close?
        mask = asdf.open(save_dir+mask_fn, lazy_load=True, copy_arrays=True)['data']['mask']
        table['alpha1'] = alpha[0]*mask # dpsidth
        table['alpha2'] = alpha[1]*mask # dpsisthdph
        compress_asdf(save_dir+alpha_fn, table, header)
        del gtlm, alpha, table
        gc.collect()

        # filtering to get shear fields
        geblm = np.zeros((2, kelm.shape[1]), dtype=kelm.dtype)
        geblm[0, :] = (gamma_filter * kelm)
        print("still complex64 = ", geblm.dtype)

        # Get the real-space shear map components using spin-2 spherical harmonics.
        t = time.time()
        gamma = ducc0.sht.experimental.synthesis(alm=geblm, lmax=lmax, spin=2, nthreads=nthreads, **geom)
        #_, gamma1, gamma2 = hp.alm2map([gtlm, gelm, gblm], new_nside, lmax=lmax, verbose=False)
        print("time getting shear = ", time.time()-t)
        print("gamma dtype = ", gamma.dtype)
        
        # Save shear
        table = {}
        table['gamma1'] = gamma[0]*mask
        table['gamma2'] = gamma[1]*mask
        compress_asdf(save_dir+gamma_fn, table, header)
        del geblm, gamma, table
        del mask
        gc.collect()
        
        sum += 1
# for measuring power spectrum
#cl_kszsq_gal = hp.anafast(cmb_fltr_masked**2, gal_masked, lmax=LMAX-1, pol=False)/fsky
