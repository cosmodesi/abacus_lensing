import sys
import time
import glob
import gc

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import treecorr
import asdf

# file name
cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_desi_lc/"
sim_name = sys.argv[1] #"AbacusSummit_base_c000_ph006" #"AbacusSummit_base_c000_ph000"
#z_s = 1.4
z_s = 1.025
z_l = 0.5
redshift_s = f"/z{z_s:.3f}/galaxies/" # galaxy lensing tracer (source)
redshift_l = f"/z{z_l:.3f}/galaxies/" # number tracer
file_name_s = cat_dir+sim_name+redshift_s+"ELGs_catalog.asdf"
file_name_l = cat_dir+sim_name+redshift_l+"ELGs_catalog.asdf"

min_sep = 0.1
max_sep = 400.
nbins = 100
bin_size = np.log(max_sep/min_sep) / nbins
s = 0.2

RA_s = asdf.open(file_name_s)['data']['RA']
DEC_s = asdf.open(file_name_s)['data']['DEC']
k_s = asdf.open(file_name_s)['data']['kappa']
g1_s = asdf.open(file_name_s)['data']['gamma1']
g2_s = asdf.open(file_name_s)['data']['gamma2']

RA_l = asdf.open(file_name_l)['data']['RA']
DEC_l = asdf.open(file_name_l)['data']['DEC']
RA_lens_l = asdf.open(file_name_l)['data']['RA_lens']
DEC_lens_l = asdf.open(file_name_l)['data']['DEC_lens']
k_l = asdf.open(file_name_l)['data']['kappa']
w_l = 1. - 2. * (2.5*s - 1.) * k_l
RAND_RA_l = asdf.open(file_name_l)['data']['RAND_RA']
RAND_DEC_l = asdf.open(file_name_l)['data']['RAND_DEC']

# load mask
z_max = np.max([z_l, z_s])
lens_save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"

mask_fns = sorted(glob.glob(lens_save_dir+f"mask_*.asdf"))
z_srcs = []
for i in range(len(mask_fns)):
    z_srcs.append(asdf.open(mask_fns[i])['header']['SourceRedshift'])
z_srcs = np.sort(np.array(z_srcs))
print("redshift sources = ", z_srcs)

mask_fn = mask_fns[np.argmin(np.abs(z_srcs - z_max))]
mask = asdf.open(mask_fn)['data']['mask']
nside = asdf.open(mask_fn)['header']['HEALPix_nside']
order = asdf.open(mask_fn)['header']['HEALPix_order']
nest = True if order == 'NESTED' else False

def get_mask_ang(mask, RA, DEC, nest, lonlat=True):
    ipix = hp.ang2pix(nside, theta=RA, phi=DEC, nest=nest, lonlat=lonlat) # RA, DEC degrees (math convention)
    return mask[ipix] == 1.

choice = get_mask_ang(mask, RA_l, DEC_l, nest) # technically new for lensed but it oche
RA_l, DEC_l, RA_lens_l, DEC_lens_l, k_l, w_l = RA_l[choice], DEC_l[choice], RA_lens_l[choice], DEC_lens_l[choice], k_l[choice], w_l[choice]
choice = get_mask_ang(mask, RAND_RA_l, RAND_DEC_l, nest)
RAND_RA_l, RAND_DEC_l = RAND_RA_l[choice], RAND_DEC_l[choice]
choice = get_mask_ang(mask, RA_s, DEC_s, nest) # technically new for lensed but it oche
RA_s, DEC_s, k_s, g1_s, g2_s = RA_s[choice], DEC_s[choice], k_s[choice], g1_s[choice], g2_s[choice]
del choice, mask; gc.collect()

#cat_s = treecorr.Catalog(file_name_s, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg', g1_col='gamma1', g2_col='gamma2', k_col='kappa')
#cat_l = treecorr.Catalog(file_name_l, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg')#, g1_col='gamma1', g2_col='gamma2', k_col='kappa') # lens
#cat_l = treecorr.Catalog(file_name_l, ra_col='RA_lens', dec_col='DEC_lens', ra_units='deg', dec_units='deg', k_col='kappa') # lens
#cat_l = treecorr.Catalog(file_name_l, ra_col='RA', dec_col='DEC', ra_units='deg', dec_units='deg', k_col='kappa') # lens
cat_s = treecorr.Catalog(ra=RA_s, dec=DEC_s, ra_units='deg', dec_units='deg', k=k_s, g1=g1_s, g2=g2_s) # source
cat_l = treecorr.Catalog(ra=RA_l, dec=DEC_l, ra_units='deg', dec_units='deg', k=k_l) # lens
cat_mag_l = treecorr.Catalog(ra=RA_l, dec=DEC_l, ra_units='deg', dec_units='deg', k=k_l, w=w_l) # lens
cat_p_l = treecorr.Catalog(ra=RA_lens_l, dec=DEC_lens_l, ra_units='deg', dec_units='deg', k=k_l) # lens
cat_mag_p_l = treecorr.Catalog(ra=RA_lens_l, dec=DEC_lens_l, ra_units='deg', dec_units='deg', k=k_l, w=w_l) # lens
cat_r_l = treecorr.Catalog(ra=RAND_RA_l, dec=RAND_DEC_l, ra_units='deg', dec_units='deg') # rand

# setup what statistics you want
gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
ng = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin') #brute=True (slows down)
kg = treecorr.KGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin') #brute=True (slows down)

rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
dd_mag_p = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
dr_mag_p = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin')
r = np.exp(gg.meanlogr)

t1 = time.time()
rr.process(cat_r_l)
dd.process(cat_l)
dr.process(cat_l, cat_r_l)
xi, varxi = dd.calculateXi(rr, dr)
dd_mag_p.process(cat_mag_p_l)
dr_mag_p.process(cat_mag_p_l, cat_r_l)
xi_mag_p, varxi_mag_p = dd_mag_p.calculateXi(rr, dr_mag_p)
np.savez(f"data/NN_zl{z_l:.3f}.npz", r=np.exp(dd.meanlogr), xi=xi, err=np.sqrt(varxi), xi_mag_p=xi_mag_p, err_mag_p=np.sqrt(varxi_mag_p))
print('Time for calculating nn correlation = ', time.time()-t1)


t1 = time.time()
gg.process(cat_s)  # Takes approx 1 minute / million objects
np.savez(f"data/GG_zs{z_s:.3f}.npz", xip=gg.xip, xim=gg.xim, r=np.exp(gg.meanlogr), err=np.sqrt(gg.varxip))
print('Time for calculating gg correlation = ', time.time()-t1)

t1 = time.time()
kg.process(cat_l, cat_s)
np.savez(f"data/KG_zl{z_l:.3f}_zs{z_s:.3f}.npz", xi=kg.xi, xi_im=kg.xi_im, r=np.exp(kg.meanlogr))
print('Time for calculatikg kg correlation = ', time.time()-t1)

t1 = time.time()
ng.process(cat_l, cat_s)
xi = ng.xi.copy()
ng.process(cat_mag_l, cat_s)
xi_mag = ng.xi.copy()
ng.process(cat_p_l, cat_s)
xi_p = ng.xi.copy()
ng.process(cat_mag_p_l, cat_s)
xi_mag_p = ng.xi.copy()

# if giving: RA, DEC and no weights
mag_p = 2.*(2.5*s - 1.) * kg.xi # mag and p
mag = 2.*(2.5*s) * kg.xi # mag
p = -2. * kg.xi # p
no_mag_p = 0. # no mag and p

# pseudo
xi_ps = xi+no_mag_p
xi_ps_mag = xi+mag
#xi_ps_p = xi+p
xi_ps_p = xi_p
#xi_ps_mag_p = xi+p+mag
xi_ps_mag_p = xi_p+mag

np.savez(f"data/NG_zl{z_l:.3f}_zs{z_s:.3f}.npz", r=np.exp(ng.meanlogr), xi=xi, xi_mag=xi_mag, xi_p=xi_p, xi_mag_p=xi_mag_p, xi_ps_mag=xi_ps_mag, xi_ps_p=xi_ps_p, xi_ps_mag_p=xi_ps_mag_p, xi_ps=xi_ps)
print('Time for calculating ng correlation = ', time.time()-t1)
quit()

# if giving: RA, DEC and no weights
#mag_p = 2.*(2.5*s - 1.) * kg.xi # mag and p
#mag = 2.*(2.5*s) * kg.xi # mag
#p = -2. * kg.xi # p
#no_mag_p = 0. # no mag and p

# if giving: RA_lens, DEC_lens and no weights
#mag_p = 2.*(2.5*s - 0.) * kg.xi # mag and p
#mag = 2.*(2.5*s + 1.) * kg.xi # mag
#p = 0. * kg.xi # p
#no_mag_p = 2. * kg.xi # no mag and p

t1 = time.time()
ng.process(cat_l, cat_s)
np.savez("data/NG.npz", xi=ng.xi+no_mag_p, xi_im=ng.xi_im, r=np.exp(ng.meanlogr), xi_mag=ng.xi+mag, xi_p=ng.xi+p, xi_mag_p=ng.xi+mag_p)
print('Time for calculating ng correlation = ', time.time()-t1)



want_plot = False
if want_plot:
    plt.plot(r, xip, color='blue')
    plt.plot(r, -xip, color='blue', ls=':')
    plt.errorbar(r[xip>0], xip[xip>0], yerr=sig[xip>0], color='blue', lw=0.1, ls='')
    plt.errorbar(r[xip<0], -xip[xip<0], yerr=sig[xip<0], color='blue', lw=0.1, ls='')
    lp = plt.errorbar(-r, xip, yerr=sig, color='blue')

    plt.plot(r, xim, color='green')
    plt.plot(r, -xim, color='green', ls=':')
    plt.errorbar(r[xim>0], xim[xim>0], yerr=sig[xim>0], color='green', lw=0.1, ls='')
    plt.errorbar(r[xim<0], -xim[xim<0], yerr=sig[xim<0], color='green', lw=0.1, ls='')
    lm = plt.errorbar(-r, xim, yerr=sig, color='green')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$ (arcmin)')

    plt.legend([lp, lm], [r'$\xi_+(\theta)$', r'$\xi_-(\theta)$'])
    plt.xlim([1, 200])
    plt.ylabel(r'$\xi_{+,-}$')
    plt.savefig("figs/xipm.png")
    plt.close()
    #plt.show()

    plt.plot(r, ng.xi*r, color='green')
    plt.plot(r, ng.xi_im*r, color='green', ls=':')
    plt.savefig("figs/gammat.png")
    plt.close()

    plt.plot(r, kg.xi*r, color='green')
    plt.plot(r, kg.xi_im*r, color='green', ls=':')
    plt.savefig("figs/kappag.png")
    plt.close()
