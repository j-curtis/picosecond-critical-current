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
		n_out[i,0] = n[i,0] - (n[i,1] - n[i,0])*v[1] - v[0]*n[i,0]*float(v[0]<0.)

		### Now for the bulk which can have unimpeded flow 
		for x in range(1,len(n[i,:])-1):
			n_out[i,x] = n[i,x] - (n[i,x+1] - n[i,x])*v[x+1] - (n[i,x] - n[i,x-1])*v[x]

		### Now for the right boundary
		n_out[i,-1] = n[i,-1] - (n[i,-1] - n[i,-2])*v[-2] + n[i,-1]*v[-1]*float(v[-1] > 0.) 

	### Now we annihilate overlapping densities
	### We do such that if n+ > n- we take n+ -> n+- n- and n- -> 0 and vice versa if n- > n+.
	#for x in range(len(n_out)):
	#	n_out[:,x] += -min(n_out[:,x])
		
	return n_out
 

### Given a discretized array of points on the domain this generates the Biot-Savart-London integro-differential equation kernel 
### Can be run once per geometry
### The operator is 
### [ -l_P/2 delta(y-y') d/dy' + 1/2pi int_-W/2^W/2 dy' (y-y')/( (y-y')^2 + z^2 ) ] J(y') 
### It is likely that a good basis for this matrix is in terms of the Chebyshev polynomials, which diagonalize the integral part of the kernel exactly
### Actually this operator acts on the link variable (currents) and returns a profile on the site variables (vortices)
### It is therefore not square 
def biot_savart_kernel(sites,links,lPearl):
	Ns = len(sites)
	Nl = len(links)
	kernel = np.zeros((Ns,Nl))
	dy = links[1]-links[0]

	### This constructs the derivative term in the bulk
	for j in range(Ns):
		kernel[j,j+1] = -0.5*lPearl/dy
		kernel[j,j] = +0.5*lPearl/dy

	### Now we construct the Biot-Savart integral part of the kernel 
	for j in range(Ns):
		site = sites[j]

		for k in range(Nl):
			link = links[k]

			kernel[j,k] += dy/(2.*np.pi) * 1./(site - link)


	return kernel 

### This method computes the inverse Biot-Savart kernel needed for the current updates 
### Can be run once per geometry 
def inverse_kernel(kernel):	

	### The kernel relates the current to the vortex density by 
	### Kernel * J = nv
	### However the current is defined on links of which there are N+1 and the vorticity is defined on sites of which there are N
	### Therefore this system is not square and we must supplement it with one more linear equation
	### This is the condition that 1/W integral dy J(y) = J(t) the total current density
	### From this we can find a square matrix and invert 
	N = kernel.shape[0]
	square_kernel = np.zeros((N+1,N+1))
	square_kernel[:-1,:] = kernel 
	square_kernel[-1,:] = 1./float(N+1)*np.ones(N+1)

	return np.linalg.inv(square_kernel)

### This method inverts the Biot-Savart kernel to find the current from the vortex dustribution and total current 
def update_current(inv_kernel,n,Jav):

	N = inv_kernel.shape[0] ### This is the number of links 

	rhs = np.zeros(N) 
	rhs[:-1] = n[0,:] - n[1,:] ### only couples the difference between density of vortices and antivortices
	
	rhs[-1] = Jav

	return inv_kernel@rhs




def main():
	W = 20.*um ### Sample width
	d = .05*um ### Sample thickness
	l = .15*um ### London penetration depth
	lPearl = 2.*l**2/d ### Pearl length 2lambda^2/d
	muv = 10.*(.04*um/ps)/Jc ### Vortex mobility, specified by drift velocity at depairing current 
	Jp = 0.05*Jc ### We take depinning current to be 5% of the depairing current 

	### Vortices live on sites and reside at -W/2 + dx, -W/2 + 2dx, ..., W/2-dx
	### Currents live on links and reside at -W/2 + dx/2, -W/2 + 3dx/2, ..., W/2-dx/2
	N = 100
	sites, links = gen_lattices(W,N)
	dx = sites[1] - sites[0]

	nts = 500
	times = np.linspace(0.,50.*ps,nts)
	dt = times[1]-times[0]

	kernel = biot_savart_kernel(sites,links,lPearl)
	invkernel = inverse_kernel(kernel)
	
	Jav = 1.5*Jp
	
	nvs = np.zeros((nts,2,len(sites))) 
	Js = np.zeros((nts,len(links)))

	nvs[0,0,N//3] = 0.1
	nvs[0,1,2*N//3] = 0.1
	Js[0,:] = update_current(invkernel,nvs[0,:,:],Jav)

	### Initial current distribution without vortices
	plt.plot(links,Js[0,:],color='purple')
	plt.axhline(Jp,color='gray',linestyle='dashed')
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$J_s/J_c$')
	plt.show()

	for i in range(1,nts):
		nvs[i,:,:] = update_vortices(nvs[i-1,:,:],Js[i-1,:],Jp,muv*dt/dx)
		Js[i,:] = update_current(invkernel,nvs[i,:,:],Jav)

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

	clrs = cm.Purples(np.linspace(0.,1.,nts))
	for i in range(nts):
		plt.plot(links,Js[i,:],color=clrs[i])
	plt.axhline(Jp,color='gray',linestyle='dashed')
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$J_s/J_c$')
	plt.show()



if __name__ == "__main__":
	main()
















