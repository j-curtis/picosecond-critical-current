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

def iext(t,ips,tps):

	return ips*np.exp(-(t/tps)**2 )


def eom(t,X,tau,ips,tps):
	phi = X[0]

	return -1./tau*np.sin(phi) + iext(t,ips,tps)


def main():
	tau = 1.*ps
	ips = 3./ps
	tps = 2.*ps

	nts = 500
	times = np.linspace(-10.*ps,10.*ps,nts)
	dt = times[1] - times[0]
	pulse = np.array([ iext(t,ips,tps) for t in times ])

	X = np.zeros((1,nts))

	sol = intg.solve_ivp(eom, (times[0],times[-1]),X[:,0],t_eval = times,args=(tau,ips,tps) )

	X[0,:] = sol.y[0,:]

	plt.plot(times,pulse,color='purple')
	plt.plot(times[1:],(X[0,1:]-X[0,:-1])/dt,color='red')
	plt.xlabel(r'$t$ [ps]')
	plt.ylabel(r'$RI/\hbar$ [THz]')
	plt.show()


if __name__ == "__main__":
	main()
















