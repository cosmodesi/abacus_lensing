import time
import gc
import glob

import healpy as hp
import asdf
import ducc0
import numpy as np
import matplotlib.pyplot as plt

sim_name = "AbacusSummit_base_c000_ph006" #sys.argv[1] #"AbacusSummit_base_c000_ph000"
save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"
lens_save_dir = f"/global/cscratch1/sd/boryanah/light_cones/{sim_name}/"

kappa_fns = sorted(glob.glob(lens_save_dir+f"kappa_00*.asdf"))
z_srcs = []
for i in range(len(kappa_fns)):
    z_srcs.append(asdf.open(kappa_fns[i])['header']['SourceRedshift'])
z_srcs = np.sort(np.array(z_srcs))
print("redshift sources = ", z_srcs)

z_source = 1089.276682
kappa_fn = kappa_fns[np.argmin(np.abs(z_srcs - z_source))]

f = asdf.open(kappa_fn, lazy_load=True, copy_arrays=True)
kappa = f['data']['kappa']
nside = f['header']['HEALPix_nside']
print("header = ", f['header'].items())

shrink = 0
if shrink:
    nside = 2048
    kappa = hp.ud_grade(kappa, nside_out=nside)#, power=2)
    np.save("kappa.npy", kappa)
    quit()
    
plot_kappa = 0
if plot_kappa:
    hp.mollview(kappa)
    plt.savefig("figs/kappa.png")
    plt.close()
    
fsky = 1.-np.mean(kappa == 0.)
print(fsky)

print("fsky == 0", fsky)
print("fsky == 0", fsky*41200.)

ell = np.arange(2*nside)
#ell = np.arange(3*nside)
lmax = ell[-1]
nthreads = 16
base = ducc0.healpix.Healpix_Base(nside, "RING")
geom = base.sht_info()
print("start kelm")
t = time.time()
kelm = ducc0.sht.experimental.adjoint_synthesis(lmax=lmax, spin=0, map=np.atleast_2d(kappa), nthreads=nthreads, **geom)
kelm *= 4*np.pi/(12*nside**2)
print("time = ", time.time()-t)
del kappa; gc.collect()
kelm = kelm.astype(np.complex128)
print(kelm.shape)

cl_kappa = hp.alm2cl(kelm.flatten())/fsky # TURNS OUT INCLUDE LMAX = LMAX IS A BUG!!!!!!!!!
np.savez(f"data/kappa_zs{z_source:.3f}.npz", cl_kappa=cl_kappa, ell=ell)
print(len(ell), len(cl_kappa))

if plot_kappa:
    plt.plot(ell, cl_kappa)
    plt.savefig("figs/cl_kappa.png")
