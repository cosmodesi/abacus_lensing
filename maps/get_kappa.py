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

from fast_cksum.cksum_io import CksumWriter
from util import histogram_hp, add_kappa_shell, add_shell
from tools import extract_steps, save_asdf, compress_asdf
from mask_kappa import get_mask
from abacusnbody.data.read_abacus import read_asdf
from abacusnbody.metadata import get_meta

# each array is 12 GB for 16384
# maybe write out at each step all 48 dudes (probably super limited by I/O)? or for each dude go through all steps?
# or for some redshift range, do the dudes and write out the rest of the dudes but that doesn't fix it
# maybe take largest mask 12.5 and save only whatever is in it for all 48 dudes, which reduces by a factor of 8
# could generate masks for each of the dudes? because when you go deeper your mask gets larger? (maube a factor of 8 is not enough or even 50)
# could also get the healpix map for each step as temporary files and just read each at each step

# adding CMB a bit add hoc
z_cmb = 1089.276682
chi_cmb = 13872.661199427605 # Mpc

# select the source redshifts (0.1 to 2.5 delta z = 0.05, around 50) tuks
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

# these are all the time steps associated with each of the healpix files
step_fns = np.zeros(len(hp_fns), dtype=int)
for i in range(len(hp_fns)):
    step_fns[i] = extract_steps(hp_fns[i])

# simulation parameters
header = asdf.open(hp_fns[0])['header']
Lbox = header['BoxSize'] # 2000. # Mpc/h
PPD = header['ppd'] # 6912
NP = PPD**3

# cosmological parameters
Om_m = header['Omega_M'] #0.315192
H0 = header['H0'] # km/s/Mpc
h = H0/100.  # 0.6736
c = 299792.458 # km/s

# comoving particle density in 1/Mpc^3
n_part = NP/(Lbox/h)**3

# all snapshots and redshifts that have light cones; early to recent redshifts
zs_all = np.load(header_dir+"redshifts.npy")
#zs_all[-1] = float("%.1f" % zs_all[-1]) # tuks needs to add redshift zero

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

# factor multiplying the standard integral (final answer should be dimensionless)
prefactor = 3.0 * H0**2 * Om_m / (2.0 * c**2)

for z_str in z_dic.keys():

    # redshift and comoving distance to the lensing sources in Mpc
    z_srcs = z_dic[z_str]
    print("current redshift sources = ", z_srcs)
    if z_srcs[-1] > z_max: # CMB
        r_ss = np.zeros_like(z_srcs)
        r_ss[:-1] = chi_of_z(z_srcs[:-1])
        r_ss[-1] = chi_cmb
    else:
        r_ss = chi_of_z(z_srcs)
    #r_ss /= h # Mpc I think bug

    # string names by which to identify masks (column names and header)
    kappa_strs = [f"Kappa_{i:05d}" for i in range(len(z_srcs))]

    # dictionary containing info about kappas
    header = {}
    header['SourceRedshifts'] = z_srcs
    header['ColumnNames'] = kappa_strs
    header['SimulationName'] = f'{sim_name}'
    header['HEALPix_nside'] = nside
    header['HEALPix_order'] = 'NESTED'
    print("header = ", header.items())
    
    # create empty array that will save our final convergence field
    #data = np.zeros((len(z_srcs), npix), dtype=np.float32)
    table = {}
    table['Kappas'] = np.zeros((len(z_srcs), npix), dtype=np.float32)

    # loop through all steps with light cone shells of interest
    for step in range(step_start, step_stop+1)[::-1]:

        # this is because our array's start corresponds to step numbers: step_start, step_start+1, step_start+2 ... step_stop
        j = step - step_min
        stepj = steps_all[j]
        # tuks
        # furthest edge of shell
        zj = zs_all[j]
        aj = 1./(1+zj)
        rj = chis_all[j]
        # mid point of shell
        zmj = z_mid[j]
        amj = 1./(1+zmj)
        rmj = chi_mid[j]
        print("z_mid, zj = ", zmj, zj)
        
        assert stepj == step, "You've messed up the counts"
        print("working with step, redshift = ", steps_all[j], zs_all[j])

        # maybe we don't have any sources at that shell redshift
        if zj > np.max(z_srcs): print("skipping redshift = ", zj); continue

        # all healpix file names which correspond to this time step
        choice_fns = np.where(step_fns == stepj)[0]
        assert (len(choice_fns) <= 3) & (len(choice_fns) > 0), "There can be at most three files in the light cones corresponding to a given step"

        # empty map corresponding to this shell (contributions from all three light cone boxes)
        rhoj = np.zeros(npix, dtype=np.float32)

        t = time.time()
        # loop through healpix files
        for choice in choice_fns:

            # first file out of 1, 2 or 3
            fn = hp_fns[choice]
            print("light cone file = ", fn)

            # load healpix data (pixel of each particle)
            f = asdf.open(fn, lazy_load=True, copy_arrays=True)
            heal = f['data']['heal'][:]
            f.close()

            # get number of particles in each pixel
            rhoj = histogram_hp(rhoj, heal)

            # alternative version (slower)
            del heal
            gc.collect()
        print("time loading and histograming = ", time.time()-t)

        
        # compute width of shell
        try:
            # distance between this shell and previous shell outwards (i.e. next shell inwards)
            drj = rj - chis_all[j+1]
            print("thickness of shell comes from zs = ", zs_all[j], zs_all[j+1])
        except:
            # works but expensive; only a problem at z = 0.1
            """
            print("need to access file to get drj")
            f = read_asdf(fn.replace('heal', 'rv')) # 1105 last for ph000
            pos = f['pos'][:]
            assert 'LightCone0' in fn # I think shouldn't happen otherwise
            dist = np.sqrt(np.sum((pos-origin)**2, axis=1))
            dist /= h # Mpc
            drj = dist.max() - dist.min()
            del f, pos, dist; gc.collect()
            """
            drj = rj - chi_of_z(z_min)
        print("drj = ",drj)
        

        # expected number of particles: delta_omega*rj**2 is the area of the pixel and drj is its depth
        dV = (delta_omega*rmj**2*drj)

        # compute analytically the mean number of particles per pixel (mean number in the box times differential volume)
        rhoj_mean = n_part*dV

        lensing_kernel = np.zeros(len(z_srcs))
        lensing_kernel[zj <= z_srcs] = ((r_ss[zj <= z_srcs] - rmj) * rmj / (amj * r_ss[zj <= z_srcs])) * drj
        """
        # load counts in asdf and compute overdensity in this shell
        t = time.time()
        rhoj = rhoj/rhoj_mean - 1.
        print("compute overdensity = ", time.time()-t)

        # add shell to kappa for each source redshift        
        for i in range(len(z_srcs)):
            zi = z_srcs[i]
            if zj > zi: continue
            
            lensing_kernel = ((r_ss[i] - rj) * rj / (aj * r_ss[i])) * drj
            table['Kappas'][i, :] += rhoj * lensing_kernel
        #table['Kappas'] += rhoj * lensing_kernel[None, :]
        """
        t = time.time()
        #data = add_shell(data, rhoj, lensing_kernel, rhoj_mean)
        table['Kappas'] = add_shell(table['Kappas'], rhoj, lensing_kernel, rhoj_mean)
        print("time adding to all sources = ", time.time()-t)
        del rhoj
        gc.collect()

    # compress table and save into asdf file
    table['Kappas'] *= prefactor
    compress_asdf(save_dir+f"kappas{z_str}.asdf", table, header)
    del table
    gc.collect()
quit()

new_nside = 8192 # to avoid overflow for healpy
new_npix = (hp.nside2npix(new_nside))

# ring2nest preload
#t = time.time()
#ipix = np.arange(npix)
#ring2nest = hp.ring2nest(nside, ipix)
#print("time ring2nest = ", time.time()-t)
#del ipix
#gc.collect()

# convert to ring (checked)
#t = time.time()
#rhoj = rhoj[ring2nest]
#print("time ring2nest = ", time.time()-t)

# downgrade to new nside
#t = time.time()
#rhoj = hp.ud_grade(rhoj, new_nside)
#print("time downgrade = ", time.time()-t)



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
    mask_src = get_mask(new_nside, chi_of_z(z_src), sim_name).astype(np.float32)
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
