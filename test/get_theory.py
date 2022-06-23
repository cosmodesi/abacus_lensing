# /global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/pyccl/tracers.py
# for the CMB tracer
# /global/homes/b/boryanah/anaconda3/envs/desc/lib/python3.7/site-packages/pyccl/boltzmann.py
# for cosmology of summit

import sys

import asdf
import pyccl as ccl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# define Cosmology object
h = 0.6736
cosmo_dic = {
    'h': h,
    'Omega_c': 0.12/h**2,
    'Omega_b': 0.02237/h**2,
    'A_s': 2.083e-9,
    'n_s': 0.9649,
    'T_CMB': 2.7255,
    'Neff': 2.0328,
    'm_nu': 0.06,
    'm_nu_type': 'single',
    'w0': -1.,
    'wa': 0.,
    'transfer_function': 'boltzmann_class'
}
cosmo = ccl.Cosmology(**cosmo_dic)

# file params
cat_dir = "/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_desi_lc/"
sim_name = sys.argv[1] #"AbacusSummit_base_c000_ph006" #"AbacusSummit_base_c000_ph000"
z_lens = 0.5
z_source = 1089.276682 #1.025 #0.95
z_min = 0.1
z_max = 2.45104189964307
redshift_s = f"/z{z_source:.3f}/galaxies/"
redshift_l = f"/z{z_lens:.3f}/galaxies/"
file_name_s = cat_dir+sim_name+redshift_s+"ELGs_catalog.asdf"
file_name_l = cat_dir+sim_name+redshift_l+"ELGs_catalog.asdf"

# map specs
nside = 16384
lmax = nside*2
ell = np.arange(lmax)

# correction neutrinos
Omega_nu = 0.0006442/h**2
Omega_m = Omega_nu + cosmo_dic['Omega_b'] + cosmo_dic['Omega_c']
Omega_cb = cosmo_dic['Omega_b'] + cosmo_dic['Omega_c']
factor = Omega_cb/Omega_m

# convergence only
cmbl_s = ccl.CMBLensingTracer(cosmo, z_source=z_source, z_min=z_min, z_max=z_max)
cls_kappa_th = ccl.angular_cl(cosmo, cmbl_s, cmbl_s, ell)/factor**2
np.savez(f"data/kappa_zs{z_source:.3f}_ccl", ell=ell, cl_kappa=cls_kappa_th)
quit()

def get_dNdz(file_name, z_edges):
    Z = asdf.open(file_name)['data']['Z']
    dNdz, _ = np.histogram(Z, bins=z_edges)
    dNdz = dNdz.astype(float)
    return dNdz
z_edges = np.linspace(0.2, 2.5, 2001)
z_cent = 0.5*(z_edges[1:] + z_edges[:-1])
dNdz_s = get_dNdz(file_name_s, z_edges)
dNdz_l = get_dNdz(file_name_l, z_edges)
bz = 0.95/ccl.background.growth_factor(cosmo, 1./(1+z_cent))


for z in [0.1, 0.3, 0.5, 0.8, 1.1, 1.7]:
    print("z = ", z)
    print("bias = ", 0.95/ccl.background.growth_factor(cosmo, 1./(1+z)))
    print("growth = ", ccl.background.growth_factor(cosmo, 1./(1+z)))
    print("--------------")

# create another weak lensing tracer with the redshift distribution of your lenses
# pass that to ccl angular with number counts tracer with same dndz (what about bias)
# get C_ell magnification term kappa,g, which I can then multiply by mag coeff 

# alpha = 5. * s / 2.; no magnification means s = 0.4
# delta_m = alpha delta_mu (magnification) = 5s delta_k; delta_p = - delta_mu (deflection) = -2 delta_k; delta_all = delta_m + delta_p; delta_mu = 2 delta_kappa
s = 0.2 # = 2 alpha / 5 = dlog10N(<m, z)/dm slope of background (source) number counts
mag_bias = (z_cent, np.ones_like(z_cent)*s)

theta = np.geomspace(0.1, 400, 100) # in arcmin, but ccl uses degrees
theta /= 60. # degrees
weak = ccl.tracers.WeakLensingTracer(cosmo, dndz=(z_cent, dNdz_s), has_shear=True, ia_bias=None, z_min=z_min, z_max=z_max)#, use_A_ia=True)
cmbl_l = ccl.CMBLensingTracer(cosmo, z_source=z_lens, z_min=z_min, z_max=z_max)
number = ccl.tracers.NumberCountsTracer(cosmo, dndz=(z_cent, dNdz_l), has_rsd=False, bias=(z_cent, bz), mag_bias=None)#, use_A_ia=True) # rsd

# compute cross power spectra
cls_shear_th = ccl.angular_cl(cosmo, weak, weak, ell)/factor**2
cls_gal_shear_th = ccl.angular_cl(cosmo, number, weak, ell)/factor
cls_gal_th = ccl.angular_cl(cosmo, number, number, ell)
cls_kappa_shear_th = ccl.angular_cl(cosmo, cmbl_l, weak, ell)/factor**2
cls_kappa_l_th = ccl.angular_cl(cosmo, cmbl_l, cmbl_l, ell)/factor**2
cls_kappa_gal_th = ccl.angular_cl(cosmo, cmbl_l, number, ell)/factor

# assessing the mag bias contribution 2 (5s/2 -1) <kappa gamma>
cls_ng_mag_p = 2.*(2.5*s - 1.) * cls_kappa_shear_th
cls_ng_mag = 2.*(2.5*s) * cls_kappa_shear_th
cls_ng_p = -2. * cls_kappa_shear_th
cls_nn_mag_p = 2. * 2.*(2.5*s) * cls_kappa_gal_th + (2.*(2.5*s))**2 * cls_kappa_l_th


print("nn modification mag bias = ", cls_nn_mag_p+cls_gal_th)
# TESTING
#cls_gal_th = ccl.angular_cl(cosmo, number, number, ell)
#number = ccl.tracers.NumberCountsTracer(cosmo, dndz=(z_cent, dNdz_l), has_rsd=False, bias=(z_cent, bz), mag_bias=mag_bias)#, use_A_ia=True) # rsd
#print("nn modification mag bias pyccl = ", cls_gal_th)

print("gal shear ad hoc bias = ", (cls_gal_shear_th + cls_ng_mag_p)[:10]) # matches perfectly with mag_bias from pyccl on the number of lensing and cmbl_l for kappa shear

xip = ccl.correlation(cosmo, ell, cls_shear_th, theta, type='GG+') # number, lensing
xim = ccl.correlation(cosmo, ell, cls_shear_th, theta, type='GG-') # number, lensing
nn = ccl.correlation(cosmo, ell, cls_gal_th, theta, type='NN') # number, number
nn_mag_p = ccl.correlation(cosmo, ell, cls_gal_th + cls_nn_mag_p, theta, type='NN') # number, number
gammat = ccl.correlation(cosmo, ell, cls_gal_shear_th, theta, type='NG') # number, lensing
gammat_mag = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_mag, theta, type='NG') # number, lensing
gammat_p = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_p, theta, type='NG') # number, lensing
gammat_mag_p = ccl.correlation(cosmo, ell, cls_gal_shear_th+cls_ng_mag_p, theta, type='NG') # number, lensing
np.savez(f"data/GG_zs{z_source:.3f}_ccl.npz", theta=theta, xip=xip, xim=xim)
np.savez(f"data/NN_zl{z_lens:.3f}_ccl.npz", theta=theta, nn=nn, nn_mag_p=nn_mag_p)
np.savez(f"data/NG_zl{z_lens:.3f}_zs{z_source:.3f}_ccl.npz", theta=theta, gammat=gammat, gammat_mag_p=gammat_mag_p, gammat_mag=gammat_mag, gammat_p=gammat_p)

want_plot = False
if want_plot:
    theta *= 60. # back to arcmin
    plt.plot(theta, xip, color='blue')
    plt.plot(theta, -xip, color='blue', ls=':')
    plt.plot(theta[xip>0], xip[xip>0], color='blue', lw=0.1, ls='')
    plt.plot(theta[xip<0], -xip[xip<0], color='blue', lw=0.1, ls='')
    lp = plt.errorbar(-theta, xip, color='blue')

    plt.plot(theta, xim, color='green')
    plt.plot(theta, -xim, color='green', ls=':')
    plt.plot(theta[xim>0], xim[xim>0], color='green', lw=0.1, ls='')
    plt.plot(theta[xim<0], -xim[xim<0], color='green', lw=0.1, ls='')
    lm = plt.errorbar(-theta, xim, color='green')

    plt.xscale('log')
    plt.yscale('log')#, nonpositive='clip')
    plt.xlabel(r'$\theta$ (arcmin)')

    plt.legend([lp, lm], [r'$\xi_+(\theta)$', r'$\xi_-(\theta)$'])
    plt.xlim( [1,200] )
    plt.ylabel(r'$\xi_{+,-}$')
    plt.savefig("figs/xipm_ccl.png")
    plt.show()

    plt.figure()
    plt.plot(theta, gammat*theta, color='blue')
    plt.savefig("figs/gammat_ccl.png")

quit()
# calculate theoretical Cls
#cmbl = ccl.CMBLensingTracer(cosmo, z_source=z_source, z_min=z_min, z_max=z_max)
#cls_cmb_th = ccl.angular_cl(cosmo, cmbl, cmbl, ell)

# convert to z (this part only helps in the cross-correlation
z_old = z.copy()
z = 1./ccl.background.scale_factor_of_chi(cosmo, chi_of_z(z)/cosmo_dic['h']) - 1.
#for i in range(len(z)): print(z[i], z_old[i])

# TESTING end

# set bias
#b = 1.72*np.ones(len(z))
b = bias/ccl.background.growth_factor(cosmo, 1./(1+z))

# create CCL tracer object for galaxy clustering
elgl = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_cent, dNdz), bias=(z,b))

# calculate theoretical Cls
cls_elg_th = ccl.angular_cl(cosmo, elgl, elgl, ell)
cls_cross_th = ccl.angular_cl(cosmo, elgl, cmbl, ell)

ccl.correlations.correlation(cosmo, ell, C_ell, theta, type='NG') # number lensing
