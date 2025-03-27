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

params = namedtuple('params','W d xi lPearl sigma Idc')

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
	return par.Idc 

### This now returns the equations of motion function for the velocity field in terms of the instaneous total current and parameters
### For total current this calls the total current function 
def eom(t,Q, sites, links, par):
	### First we construct the sites and links and the relevant lattices 

	Ns = len(sites)
	Nl = len(links)
	dy = links[1]-links[0]

	BSkernel = np.zeros((Ns,Ns)) ### The matrix is rendered square with the first N entries the derivative and integral kernels and the last entry the total current constraint
	
	BSkernel[-1,:] = np.ones(Ns)*dy  ### Total current constraint
	BSkernel[:-1,:] = dy/( np.tensordot(links,np.ones_like(sites),axes=-1)  - np.tensordot(np.ones_like(links),sites,axes=-1) )


	rhs = np.zeros(Ns)

	rhs[-1] = -Itot(t,par)

	for i in range(Ns-1):
		rhs[i] = -2.*np.pi/dy*( Q[i+1] - Q[i]) ### This should give finite element derivative for Q on the links 

	eom = -1./(2.*par.d*par.sigma)*np.linalg.inv(BSkernel)@rhs ### This inverts the Biot-Savart kernel and divides by conductivity 
	eom += -1./(par.d*par.sigma*par.lPearl)*Q[:]*(np.ones_like(Q) - (2.*np.pi*par.xi/phi0)**2*Q[:]**2) 

	return eom 

### This will solve the equations of motion for a given set of times and parameters 
def solve_eom(sites,links,times,par):

	Ns = len(sites)
	Nl = len(links)
	dy = links[1]-links[0]

	Q0 = np.zeros_like(sites)

	t0 = times[0]
	tf = times[-1]

	sol = intg.solve_ivp(eom,(t0,tf),Q0,t_eval=times,args=(sites,links,par))
	print(sol.t.shape)

	return sol



def main():

	W = 20.*um ### Sample width
	d = .05*um ### Sample thickness
	xi = .002*um ### Coherence length is very short 
	l = .15*um ### London penetration depth
	lPearl = 2.*l**2/d ### Pearl length 2lambda^2/d
	sigma = 10. ### Check units 
	Idc = 300.*2.*np.pi*phi0 ### Check units 
	print(Idc)

	param = params(W,d,xi,lPearl,sigma,Idc)

	N = 40
	sites, links = gen_lattices(W,N)

	nts = 1000
	times = np.linspace(0.,100.*ps,nts)

	Q0 = np.zeros_like(sites)

	t0 = times[0]
	tf = times[-1]

	sol = intg.solve_ivp(eom,(t0,tf),Q0,t_eval=times,args=(sites,links,param))
	print(sol.message)

	plt.plot(sites, 2.*np.pi*xi*sol.y[:,-1]/phi0)
	plt.xlabel(r'$y$ [$\mu$m]')
	plt.ylabel(r'$Q$ [$\phi_0/\xi$]')
	plt.show()







if __name__ == "__main__":
	main()











