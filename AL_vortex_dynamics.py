### Picosecond dynamics of vortices in a current biased thin wide film of cuprate
### Jonathan Curtis 
### 02/28/25

import numpy as np
from scipy import integrate as intg
import time
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as mclr


### Plotting settings 
#plt.rc('figure', dpi=100)
#plt.rc('figure',figsize=(4,1.7))
plt.rc('font', family = 'Times New Roman')
plt.rc('font', size = 14)
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('axes', labelsize=18)
plt.rc('lines', linewidth=2.5)


um = 1. ### Length is measured in microns
ps = 1. ### Time is measured in ps 
xi = 2.e-3 *um ### We use coherence length as the short-distance cutoff length 
Jc = 1. ### We measure current density in units of the depairing current density Jc 

### This gives the relation between drift velocity and superfluid current density as a function of current
### This is a nonlinear function in the presence of pinning and this model is phenomenologically input here 
@np.vectorize
def depinning_function(Js,Jp):
	if np.abs(Js) < Jp:
		return 0.

	else:
		return Js - np.sign(Js)*Jp

### This computes the vortex drift velocity as a function of the local current and the density
### This needs to be multiplied by the charge for each species still (i.e. this assumes positive vorticity)
def vd(Js,Jp,muv):
	return muv * depinning_function(Js,Jp)

### We define a site lattice where vortex density resides and a link lattice where currents live
### The boundaries are located at -W/2, W/2 and we control the lattice by specifying the number of steps N
### The boundaries are actually links (as they control the currents) and therefore this is all half unit cell offset
### This will be the number of sites in the site lattice which will have points -W/2 + dx/2, ... W/2 -dx/2 where (N+1)dx = W 
def gen_lattices(W,N):
	### First we generate the lattice for sites where currents live, which are links
	links = np.linspace(-W/2,W/2.,N+1) 

	### Site lattice is taken by translating by half a lattice step and discarding the last point 
	sites = np.linspace(-W/2,W/2.,N+1)
	sites += (sites[1] -sites[0])/2.
	sites = sites[:-1]

	return sites,links

### We model separately the dynamics of vortices and antivortices
### We pass the two densities as [n+,n-]
### muvdtdx = muv*dt/dx and is redefined to include the lattice constant and time step 

### We use the model for dynamics that they are independently conserved except when they overlap
### In such case, if there are n+ vortices and n- antivortices on the same position then we will annihilate them so that only the net density is remaining

### The method will consist of two steps
### First it will flow each of the densities independently according to the continuity equations
### To handle the boundaries we will impose that the vortex flux at the boundaries can only be outwards (below critical current)
### In the second step we will annihilate excess densities on overlapping sites 
def update_vortices(n,Js,Jp,muvdtdx):
	### First we flow according to the vortex convection
	### We will return a new array of vortex profiles 
	n_out = np.zeros_like(n)

	for i in range(2):
		### We compute the vortex velocities first 
		s = (-1)**i ### We do first vortex (s= 1) then antivortex (s=-1)
		v = s*vd(Js,Jp,muvdtdx)

		### Now we integrate the continuity equation  
		### First we do the left end point
		### We only allow vortices to flow outwards at the boundary 
		### Fictious boundary densities 
		nL = 0.
		nR = 0.
		n_out[i,0] = n[i,0] - (n[i,1] - n[i,0])*v[1] - (n[i,0]- nL)*float(v[0]<0.)

		### Now for the bulk which can have unimpeded flow 
		for x in range(1,len(n[i,:])-1):
			n_out[i,x] = n[i,x] - (n[i,x+1] - n[i,x])*v[x+1] - (n[i,x] - n[i,x-1])*v[x]

		### Now for the right boundary
		n_out[i,-1] = n[i,-1] - (nR - n[i,-1])*v[-1]*float(v[-1] > 0.) - (n[i,-1] - n[i,-2])*v[-2] 

	### Now we annihilate overlapping densities
	### We do such that if n+ > n- we take n+ -> n+- n- and n- -> 0 and vice versa if n- > n+.
	#for x in range(len(n_out)):
	#	n_out[:,x] += -min(n_out[:,x])
		
	return n_out
 

### Given a discretized array of points on the domain this generates the Biot-Savart integral equation kernel 
### Can be run once per geometry
def biot_savart_kernel(xpts):
	N = len(x)
	kernel = np.zeros((N,N))
	dx = xpts[1]-xpts[0]

	z = xi

	for j in range(N):
		x = xpts[j]
		for k in range(N):
			y = xpts[k]

			kernel[j,k] = 1./(2.*np.pi) * (x - y)/( (x -y)**2 + z**2 ) 


	return kernel*dx 


### This is the full kernel for the current profile including both Biot-Savart and London equations 
def current_kernel(xpts,l_Pearl):
	N = len(x)
	kernel = biot_savart_kernel(xpts)
	dx = xpts[1]-xpts[0]

	for j in range(1,N):
		kernel[j,j-1] += - l_Pearl/dx 
		kernel[j,j] += l_Pearl/dx  

	return kernel


def main():
	W = 20.*um ### Sample width
	d = .05*um ### Sample thickness
	l = .15*um ### London penetration depth
	lP = 2.*l**2/d ### Pearl length 2lambda^2/d
	muv = (.04*um/ps)/Jc ### Vortex mobility, specified by drift velocity at depairing current 
	Jp = 0.05*Jc ### We take depinning current to be 5% of the depairing current 

	### Vortices live on sites and reside at -W/2 + dx, -W/2 + 2dx, ..., W/2-dx
	### Currents live on links and reside at -W/2 + dx/2, -W/2 + 3dx/2, ..., W/2-dx/2
	N = 10
	sites, links = gen_lattices(W,N)
	dx = sites[1] - sites[0]

	### Supercurrent density profile 
	### Supercurrent lives on the links 
	Js = 100.*np.cosh(links*2./W)/np.cosh(1.)*Jp

	plt.plot(links,Js,color='purple',marker='.')
	plt.axhline(Jp,color='gray',linestyle='dashed')
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$J_s/J_c$')
	plt.ylim(0.,2.*max(Js))
	plt.show()

	nts = 2000
	times = np.linspace(0.,100.*ps,nts)
	dt = times[1]-times[0]
	nvs = np.zeros((nts,2,len(sites)))

	nvs[0,0,:] = np.maximum(-0.01*np.sin(2.*np.pi*sites/W),0.)
	nvs[0,1,:] = np.maximum(0.01*np.sin(2.*np.pi*sites/W),0.)

	for i in range(1,nts):
		nvs[i,:,:] = update_vortices(nvs[i-1,:,:],Js,Jp,muv*dt/dx)

	clrs = cm.Blues(np.linspace(0.3,1.,nts))

	for i in range(nts):
		plt.plot(sites,nvs[i,0,:],color=clrs[i])
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$n_+$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()

	clrs = cm.Reds(np.linspace(0.3,1.,nts))
	for i in range(nts):
		plt.plot(sites,nvs[i,1,:],color=clrs[i])
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$n_-$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()


	clrs = cm.coolwarm(np.linspace(0.,1.,nts))
	for i in range(nts):
		plt.plot(sites,nvs[i,0,:] - nvs[i,1,:],color=clrs[i])
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$n_v$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()

	plt.plot(times,np.sum(nvs[:,0,:],axis=-1))
	plt.xlabel(r'$t$ [ps]')
	plt.ylabel(r'$\langle n_+\rangle$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()




if __name__ == "__main__":
	main()
















