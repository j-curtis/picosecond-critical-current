### Picosecond dynamics of London current density in thin-film including dynamic screening and normal fluid
### Jonathan Curtis 
### 03/27/25

import numpy as np
from scipy import integrate as intg
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr
from collections import namedtuple 

params = namedtuple('params','W d xi lab sigma2D tauGL Idc Ips t0 td')

### Plotting settings 
#plt.rc('figure', dpi=100)
#plt.rc('figure',figsize=(4,1.7))
#plt.rc('font', family = 'Times New Roman')
#plt.rc('font', size = 14)
#plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=14)
#plt.rc('ytick', labelsize=14)
#plt.rc('axes', labelsize=18)
#plt.rc('lines', linewidth=2.5)


um = 1. ### Length is measured in microns
ps = 1. ### Time is measured in ps 
phi0 = 1. ### We measure fields and currents in units of flux quantum  

mT = .4836* phi0/um**2 ### This is millitesla in units of flux quantum and microns
mA = 1.2566*um*mT ### This converts from mT to mA assuming mu0 = 1 units 
mV = .2418/ps ### millivolts in units of inverse THz = h*1THz/e

### We define a site lattice where current density resides and a link lattice where gradients live
### These have coordinates sites: -W/2 + n dy with n = 0,1,2,...N and links: l_m = -W/2 + (m+1/2)dy with m = 0,1,2,...N-1
### There are therefore N+1 sites and N links 
### Boundaries are included in sites 
def gen_lattices(W,N):
	### First we generate the lattice for sites where currents live
	sites = np.linspace(-W/2,W/2.,N+1) 

	### link lattice is taken by translating by half a lattice step and discarding the last point 
	links = np.linspace(-W/2,W/2.,N+1)
	links += (sites[1] -sites[0])/2.
	links = links[:-1]

	return sites,links

### This returns the total current as a function of time -- for the moment we use a simple static total current 
def Itot(t, par):
	return par.Idc + par.Ips*np.exp(-(t-par.t0)**2/(par.td)**2)

### This now returns the equations of motion function for the velocity field in terms of the instaneous total current and parameters
### For total current this calls the total current function 
### The vector X = [Q, f] where Q is the superfluid velocity and f is the amplitude of the order parameter field 
### These both live on the sites
def eom(t,X, sites, links, par):
	### First we construct the sites and links and the relevant lattices 

	Ns = len(sites)
	Nl = len(links)
	dy = links[1]-links[0]

	### Now we extract the Q and f degrees of freedom
	Q = X[:Ns]
	f = X[Ns:]

	out = np.zeros_like(X)

	### This evolves the amplitude 
	### It has a nonlinear local part and a diffusion term 
	dfdy = np.gradient(f,dy)
	d2fdy2 = np.gradient(dfdy,dy)
	out[Ns:] = 1./par.tauGL*(np.ones_like(f) - f**2 - (2.*np.pi*par.xi/phi0)**2 * Q**2 )*f + par.xi**2/par.tauGL *d2fdy2 ### This is the TDGL in the y dimension


	### To evolve the velocities we must first obtain the current density by inverting the Biot Savart law
	BSkernel = np.zeros((Ns,Ns)) ### The matrix is rendered square with the first N entries the derivative and integral kernels and the last entry the total current constraint
	
	BSkernel[-1,:] = np.ones(Ns)*dy  ### Total current constraint
	BSkernel[:-1,:] = (dy/(2.*np.pi) )/( np.tensordot(links,np.ones_like(sites),axes=-1)  - np.tensordot(np.ones_like(links),sites,axes=-1) )

	rhs = np.zeros(Ns)

	for i in range(Nl):
		rhs[i] = (Q[i+1]-Q[i])/dy ### This should give finite element derivative for Q on the links
		### Importantly this has size N_s - 1 and therefore cannot simply employ the numpy gradient method
	
	rhs[-1] = Itot(t,par)

	out[:Ns] = 1./(par.sigma2D)*np.linalg.inv(BSkernel)@rhs ### This inverts the Biot-Savart kernel and divides by conductivity 
	out[:Ns] += -par.d/(par.sigma2D*par.lab**2)*Q[:]*f[:]**2

	return out 

### This will solve the equations of motion for a given set of times and parameters 
def solve_eom(sites,links,times,par):

	Ns = len(sites)
	Nl = len(links)
	dy = links[1]-links[0]

	Q0 = np.zeros_like(sites)
	f0 = np.ones_like(sites)

	X0 = np.concatenate((Q0,f0))

	t0 = times[0]
	tf = times[-1]

	sol = intg.solve_ivp(eom,(t0,tf),X0,t_eval=times,args=(sites,links,par))

	return sol

### This will compute the average electric vs time given the supercurrent 
### Q is passed as an array of the form [y,t]
def voltage(times,Q):
	qav = np.mean(Q,axis=0)
	dt = times[1] - times[0]

	return np.gradient(qav,dt)

def main():

	W = 10.*um ### Sample width
	d = .05*um ### Sample thickness
	L = 20.*um ### Sample length
	xi = 0.01*um ### Coherence length is very short 
	tauGL = 1.*ps ### GL relaxation time 
	lab = .15*um ### London penetration depth in ab plane
	lPearl = 2.*lab**2/d ### Pearl length 2lambda^2/d
	sigma2D = 0.2 ### 2D conductivity is unitless

	Idc = 170*mA 
	Ips = 0.*mA 
	t0 = 0.*ps 
	td = 3.*ps 

	param = params(W,d,xi,lab,sigma2D,tauGL,Idc,Ips,t0,td)

	N = 20
	sites, links = gen_lattices(W,N)

	nts = 200
	times = np.linspace(-100.*ps,30.*ps,nts)

	sol = solve_eom(sites,links,times,param)

	inds = np.array(list(range(0,nts,50)))
	clrs = cm.Purples(np.linspace(0.2,1.,len(inds)))

	for i in range(len(inds)):
		plt.plot(sites, sol.y[len(sites):,inds[i]],color=clrs[i])
	plt.xlabel(r'$y$ [$\mu$m]')
	#plt.ylabel(r'Qf$ [mA]')
	plt.ylabel(r'$f$')
	plt.show()

	for i in range(len(inds)):
		plt.plot(sites, sol.y[:len(sites),inds[i]]/mA,color=clrs[i])
	plt.xlabel(r'$y$ [$\mu$m]')
	plt.ylabel(r'$Q$ [mA]')
	#plt.ylabel(r'$f$')
	plt.show()

	E = voltage(sol.t,sol.y)

	plt.plot(times, E[:]*L/mV)
	plt.xlabel(r'$t$ [ps]')
	plt.ylabel(r'$V$ [mV]')
	plt.show()






if __name__ == "__main__":
	main()











