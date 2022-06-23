import os
import gc
import sys
import glob
from pathlib import Path
import argparse

import numpy as np
import healpy as hp
import asdf
from astropy.io import fits
from astropy.io import ascii
from scipy.interpolate import interp1d

from generate_randoms import gen_rand
sys.path.append("/global/homes/b/boryanah/abacus_lensing/maps")
from tools import compress_asdf

DEFAULTS = {}
DEFAULTS['sim_name'] = "AbacusSummit_base_c000_ph006" # "AbacusSummit_huge_c000_ph201"#"AbacusSummit_base_c000_ph000" # "AbacusSummit_huge_c000_ph201"

def get_norm(gals_pos, origin):
    """ get normal vector and chis"""
    gals_norm = gals_pos - origin
    gals_chis = np.linalg.norm(gals_norm, axis=1)
    assert len(gals_pos) == len(gals_chis)
    gals_norm /= gals_chis[:, None]
    gals_min = np.min(gals_chis)
    gals_max = np.max(gals_chis)

    return gals_norm, gals_chis, gals_min, gals_max


def get_ra_dec_chi(norm, chis):
    """ given normal vector and chis, return, ra, dec, chis"""
    theta, phi = hp.vec2ang(norm)
    ra = phi
    dec = np.pi/2. - theta
    ra *= 180./np.pi
    dec *= 180./np.pi
    
    return ra, dec, chis

def relate_chi_z(sim_name):
    # load zs from high to low
    data_path = Path("/global/homes/b/boryanah/repos/abacus_lc_cat/data_headers/")

    # all redshifts, steps and comoving distances of light cones files; high z to low z
    zs_all = np.load(data_path / sim_name / "redshifts.npy")
    chis_all = np.load(data_path / sim_name / "coord_dist.npy")
    zs_all[-1] = float('%.1f'%zs_all[-1])

    # get functions relating chi and z
    chi_of_z = interp1d(zs_all,chis_all)
    z_of_chi = interp1d(chis_all,zs_all)
    return chi_of_z, z_of_chi

def read_fits(gal_fn, cat_dir, tracer, redshift, sim_name):
    # load file with catalog
    hdul = fits.open(Path(cat_dir) / tracer / (f"z{redshift:.3f}") / sim_name / (gal_fn), allow_pickle=True)
    assert hdul[1].header['Gal_type'.upper()] == tracer
    gals_pos = np.vstack((hdul[1].data['x'], hdul[1].data['y'], hdul[1].data['z'])).T
    N_gals = gals_pos.shape[0]
    print("galaxy number loaded = ", N_gals)
    return gals_pos

def read_dat(want_rsd, cat_dir, tracer, redshift, sim_name):
    # load file with catalog
    rsd_str = '_rsd' if want_rsd else ''
    f = ascii.read(Path(cat_dir) / sim_name / (f"z{redshift:.3f}") / f'galaxies{rsd_str}' / f'{tracer}s.dat')
    gals_pos = np.vstack((f['x'], f['y'], f['z'])).T
    N_gals = gals_pos.shape[0]
    print("galaxy number loaded = ", N_gals)
    return gals_pos

def main(sim_name, want_rsd=False):
    # parameter choices
    #redshifts = [0.400, 0.450, 0.500, 0.575, 0.650, 0.725, 0.800, 0.875, 0.950, 1.025, 1.100, 1.175, 1.250, 1.325, 1.400]
    redshifts = [1.325, 1.400] #[0.400, 0.500, 0.650, 0.800, 0.950]
    tracer = "ELG"
    rands_fac = 20
    
    # immutables
    sim_dir = "/global/project/projectdirs/desi/cosmosim/Abacus/halo_light_cones/"
    lens_save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"
    offset = 10. # Mpc/h
    
    # read from simulation header
    header = asdf.open(f"{sim_dir}/{sim_name}/z{redshifts[0]:.3f}/lc_halo_info.asdf")['header']
    Lbox = header['BoxSizeHMpc']
    mpart = header['ParticleMassHMsun'] # 5.7e10, 2.1e9
    origins = np.array(header['LightConeOrigins']).reshape(-1,3)
    origin = origins[0]
    print(f"mpart = {mpart:.2e}")
    
    # fits file with galaxy mock
    #cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_lc_output/catalogs/LightCone/"
    #gal_fn = "catalog_rsd_xi2d_lrg_main_z0.5_velbias_B_s_skip60_mockcov.fits"

    # standard dat file with galaxy mock
    cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_desi_lc/"
    if want_rsd:
        gal_fn = f"galaxies_rsd/{tracer}s.dat"
        rsd_str = "_rsd"
    else:
        gal_fn = f"galaxies/{tracer}s.dat"
        rsd_str = ""

    # functions relating chi and z
    chi_of_z, z_of_chi = relate_chi_z(sim_name)

    # nside
    import glob
    gamma_fns = sorted(glob.glob(lens_save_dir+f"gamma_*.asdf"))
    z_srcs = []
    for i in range(len(gamma_fns)):
        z_srcs.append(asdf.open(gamma_fns[i])['header']['SourceRedshift'])
    z_srcs = np.sort(np.array(z_srcs))
    print("redshift sources = ", z_srcs)
    z_srcs = np.unique(z_srcs)
    
    nside = asdf.open(gamma_fns[2])['header']['HEALPix_nside']
    order = asdf.open(gamma_fns[2])['header']['HEALPix_order']
    print("nside and order = ", nside, order)
    
    for redshift in redshifts:        
        if '.fits' in gal_fn:
            gals_pos = read_fits(gal_fn, cat_dir, tracer, redshift, sim_name)
        elif '.dat' in gal_fn:
            gals_pos = read_dat(want_rsd, cat_dir, tracer, redshift, sim_name)

        # get the unit vectors and comoving distances to the observer
        gals_norm, gals_chis, gals_min, gals_max = get_norm(gals_pos, origin)
        print("closest and furthest distance of gals = ", gals_min, gals_max)

        # directory for saving stuff
        save_dir = Path(cat_dir) / sim_name / (f"z{redshift:.3f}") / f'galaxies{rsd_str}'

        # generate randoms in L shape
        rands_pos, rands_norm, rands_chis = gen_rand(len(gals_chis), gals_min, gals_max, rands_fac, Lbox, offset, origins)
        
        # convert the unit vectors into RA and DEC
        RA, DEC, CZ = get_ra_dec_chi(gals_norm, gals_chis)
        rands_RA, rands_DEC, rands_CZ = get_ra_dec_chi(rands_norm, rands_chis)

        # convert chi to redshift
        Z = z_of_chi(CZ)
        rands_Z = z_of_chi(rands_CZ)

        # dictionary containing value added fields
        table = {}
        table['Z'] = Z
        table['RA'] = RA
        table['DEC'] = DEC
        table['CZ'] = CZ
        table['gamma1'] = np.zeros(len(RA))
        table['gamma2'] = np.zeros(len(RA))
        table['kappa'] = np.zeros(len(RA))
        table['RA_lens'] = np.zeros(len(RA))
        table['DEC_lens'] = np.zeros(len(RA))
        table['RAND_RA'] = rands_RA
        table['RAND_DEC'] = rands_DEC
        table['RAND_Z'] = rands_Z
        table['RAND_CZ'] = rands_CZ
        table['RAND_gamma1'] = np.zeros(len(rands_RA))
        table['RAND_gamma2'] = np.zeros(len(rands_RA))
        table['RAND_kappa'] = np.zeros(len(rands_RA))
        header = {}
        header['SimulationName'] = sim_name
        header['GalaxyTracer'] = tracer
        header['CatalogRedshift'] = redshift
        if want_rsd:
            header['RSD'] = 'ON'
        else:
            header['RSD'] = 'OFF'

        # scope of redshifts (randoms may have a wider range)
        Z_min = np.min(rands_Z)
        Z_max = np.max(rands_Z)
        print("rand Z min/max", Z_min, Z_max)
        Z_min = np.min([Z_min, np.min(Z)])
        Z_max = np.max([Z_max, np.max(Z)])
        print("overall Z min/max", Z_min, Z_max)
        i_zsrc_min = np.argmin(np.abs(Z_min - z_srcs))
        i_zsrc_max = np.argmin(np.abs(Z_max - z_srcs))
        print("RA min/max", RA.min(), RA.max())
        print("DEC min/max", DEC.min(), DEC.max())
        
        # convert angles to pixel numbers
        nest = True if order == 'NESTED' else False
        #nside = 2048 # TESTING!!!!!!!!!!!
        print("should be false = ", nest)
        ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=True) # RA, DEC degrees (math convention)
        rands_ipix = hp.ang2pix(nside, theta=rands_RA, phi=rands_DEC, nest=nest, lonlat=True) # RA, DEC degrees (math convention)

        # go into spherical coordinates
        THETA = (90.-DEC)*np.pi/180. # checked in Lewis Piranya 2008
        PHI = RA*np.pi/180.
        
        # for each galaxy find the closest zsrc
        sum = 0
        rands_sum = 0
        delta_z = z_srcs[1]-z_srcs[0]
        print("delta_z = ", delta_z)
        print("z sources = ", z_srcs)
        print("total galaxy number = ", len(Z))
        for i in range(len(z_srcs)):
            if i < i_zsrc_min or i > i_zsrc_max: continue
            print("source redshift = ", z_srcs[i])
            
            # select galaxies corresponding to that redshift source
            choice = (np.abs(Z - z_srcs[i]) <= delta_z/2.)
            rands_choice = (np.abs(rands_Z - z_srcs[i]) <= delta_z/2.)
            sum += np.sum(choice)
            print("sum = ", sum)
            rands_sum += np.sum(rands_choice)
            ipix_choice = ipix[choice]
            rands_ipix_choice = rands_ipix[rands_choice]

            # save gamma and kappa values for galaxies and randoms
            #i = 9 # TESTING 0.55
            print(asdf.open(f"{lens_save_dir}/gamma_{i:05d}.asdf")['header'].items())
            gamma1 = asdf.open(f"{lens_save_dir}/gamma_{i:05d}.asdf")['data']['gamma1']
            #gamma1 = np.load("../test/gamma1_nside.npy") # TESTING!!!!!!!!!!!
            table['gamma1'][choice] = gamma1[ipix_choice]
            table['RAND_gamma1'][rands_choice] = gamma1[rands_ipix_choice]
            del gamma1; gc.collect()
            gamma2 = asdf.open(f"{lens_save_dir}/gamma_{i:05d}.asdf")['data']['gamma2']
            #gamma2 = np.load("../test/gamma2_nside.npy") # TESTING!!!!!!!!!!!
            table['gamma2'][choice] = -gamma2[ipix_choice] # minus sign is important
            table['RAND_gamma2'][rands_choice] = -gamma2[rands_ipix_choice]  # minus sign is important
            del gamma2; gc.collect()
            kappa = asdf.open(f"{lens_save_dir}/kappa_{i:05d}.asdf")['data']['kappa']
            #kappa = np.load("../test/kappa.npy") # TESTING!!!!!!!!!!!
            table['kappa'][choice] = kappa[ipix_choice]
            table['RAND_kappa'][rands_choice] = kappa[rands_ipix_choice]
            del kappa; gc.collect()

            # call the gamma and alpha maps
            alpha1 = asdf.open(f"{lens_save_dir}/alpha_{i:05d}.asdf")['data']['alpha1']
            alpha2 = asdf.open(f"{lens_save_dir}/alpha_{i:05d}.asdf")['data']['alpha2']
            
            # derived quantities from the deflection angle
            alpha = np.sqrt(alpha1**2+alpha2**2)
            cosdelta = alpha1/alpha
            sindelta = alpha2/alpha
            del alpha1, alpha2; gc.collect()
            
            # save deflected positions following 5.1 in Fosalba 2013
            alp = alpha[ipix_choice]
            del alpha; gc.collect()
            sdel = sindelta[ipix_choice]
            del sindelta; gc.collect()
            cdel = cosdelta[ipix_choice]
            del cosdelta; gc.collect()
            DPHI = np.arcsin(np.sin(alp)*sdel/np.sin(THETA[choice]))
            THETAP = np.arccos(np.cos(alp)*np.cos(THETA[choice])-np.sin(alp)*np.sin(THETA[choice])*cdel)
            del alp, sdel, cdel; gc.collect()
            table['RA_lens'][choice] = (PHI[choice]+DPHI)*180./np.pi
            table['DEC_lens'][choice] = (np.pi/2. - THETAP)*180./np.pi
            del THETAP, DPHI; gc.collect()
            
        assert len(Z) == sum
        assert len(rands_Z) == rands_sum
        
        # compress table and save into asdf file
        compress_asdf(save_dir / f"{tracer}s_catalog.asdf", table, header)
        
         
        #hdu = fits.PrimaryHDU(export_array)
        #hdulist = fits.HDUList([hdu])
        #hdulist.writeto(out_file_name)
        #hdulist.close()
        
        fns = list(save_dir.glob("*.asdf"))
        for fn in fns:
            os.chmod(fn, 0o755)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--sim_name', help='Simulation name', default=DEFAULTS['sim_name'])
    parser.add_argument('--want_rsd', help='Include RSD effects', action='store_true')

    args = vars(parser.parse_args())
    main(**args)
