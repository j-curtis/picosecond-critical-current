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


um = 1. ### Lengths will be ultimately referenced in microns
ps = 1. ### Time is measured in ps 
xi = 2.e-3 *um ### We use coherence length as the short-distance cutoff length 
Jcrit = 1. ### We measure current density in units of the critical current density 


### We model separately the dynamics of vortices and antivortices
### We use the model for dynamics that they are independently conserved except when they interact
### If there are n+ vortices and n- antivortices on the same position then we will annihilate them so that only the net density is remaining
### At the boundaries we will model the exit of vortices by a model where the vortices can enter only if J = Jc but can leave at any current

### This gives the I-V characteristic for the current in the presence of depinning current threshold
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


### This function will implement one time step update for the vortex densities
### First it will flow each of the densities independently
### Then it will annihilate the excess density on overlapping sites 
### We also implement absorbing boundary conditions by allowing vortices to flow over the edge, and then simply annhilating those that do
### We pass the two densities as [n+,n-]
### muvdtdx = muv*dt/dx and is redefined to include the lattice constant and time step 
def update_vortices(n,Js,Jp,muvdtdx):
	### First we flow according to the vortex convection
	### We will return a new array of vortex profiles 
	n_out = np.zeros_like(n)

	for i in range(2):
		### We compute the vortex velocities first 
		s = (-1)**i ### We do first vortex (s= 1) then antivortex (s=-1)
		v = s*vd(Js,Jp,muvdtdx)

		### Now we compute the current 
		j = n[i,:]*v[:]
		### Now we compute the divergence 
		### We use a symmetric derivative in the bulk
		for x in range(1,len(Js)-1):
			n_out[i,x] = n[i,x] - (j[x+1] - j[x-1])/2.
		## For the end points we introduce fictitous currents which are zero
		n_out[i,0] = n[i,0] - (j[1] - j[0])
		n_out[i,-1] = n[i,-1] - (j[-1]-j[-2])

	### Now we annihilate overlapping densities
	### We do such that if n+ > n- we take n+ -> n+- n- and n- -> 0 and vice versa if n- > n+.
	for x in range(len(Js)):
		n_out[:,x] += -min(n_out[:,x])
		
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
	muv = (.04*um/ps)/Jcrit ### Vortex mobility, specified by drift velocity at depairing current 
	Jp = 0.05*Jcrit ### We take depinning current to be 5% of the depairing current 

	dx = 1.*um
	xpts = np.arange(-W/2.,W/2.+dx,dx)
	nxs = len(xpts)

	### Supercurrent density profile 
	Js = (np.cosh(xpts*2./W)-1.)/(np.cosh(1.)-1.)*40.*Jp

	plt.plot(xpts,Js,color='purple')
	plt.axhline(Jp,color='gray',linestyle='dashed')
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$J_s/J_c$')
	plt.ylim(0.,2.*max(Js))
	plt.show()

	nts = 30
	times = np.linspace(0.,300.*ps,nts)
	dt = times[1]-times[0]
	nvs = np.zeros((nts,2,nxs))

	nvs[0,0,:] = np.maximum(0.1*np.sign(xpts),0.)
	nvs[0,1,:] = np.maximum(-0.1*np.sign(xpts),0.)

	for i in range(1,nts):
		nvs[i,:,:] = update_vortices(nvs[i-1,:,:],Js,Jp,muv*dt/dx)

	clrs = cm.Blues(np.linspace(0.3,1.,nts))

	for i in range(nts):
		plt.plot(xpts,nvs[i,0,:],color=clrs[i])
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$n_+$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()

	clrs = cm.Reds(np.linspace(0.3,1.,nts))
	for i in range(nts):
		plt.plot(xpts,nvs[i,1,:],color=clrs[i])
	plt.xlabel(r'$x$ [$\mu$m]')
	plt.ylabel(r'$n_-$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()

	plt.plot(times,np.sum(nvs[:,0,:],axis=-1))
	plt.xlabel(r'$t$ [ps]')
	plt.ylabel(r'$\langle n_+\rangle$ [$\phi_0/\mu$m$^{2}$]')
	plt.show()




if __name__ == "__main__":
	main()
















