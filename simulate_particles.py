import sys
import os
import time
import numpy as n
import argparse
import math
import resource

import multiprocessing as mp

sys.path.append('cryoem/')
sys.path.append('cryoem/util')

from cryoem.cryoio import ctf
from cryoem.cryoio import mrc

from cryoem.util import format_timedelta

from cryoem import cryoem
from cryoem import geom
from cryoem import cryoops
from cryoem import density
from cryoem import sincint

import numpy.fft as fourier

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# For parallel https://stackoverflow.com/questions/15639779/why-does-multiprocessing-use-only-a-single-core-after-i-import-numpy
os.environ["OPENBLAS_MAIN_FREE"] = "1"

# Set the files open limit (must exceed the simulation chunk size)
resource.setrlimit(resource.RLIMIT_NOFILE, (1100, 1100))

# matplotlib configuration
mpl.rcParams['figure.dpi'] = 100
plt.style.use(['dark_background'])

def main(args):
	# setup microscope and ctf parameters
	params = {}
	params['defocus_min'] = 10000
	params['defocus_max'] = 20000
	params['defocus_ang_min'] = 0
	params['defocus_ang_max'] = 360
	params['accel_kv'] = 300
	params['amp_contrast'] = 0.07
	params['phase_shift'] = 0
	scale = 1
	params['spherical_abberr'] = 2.7
	params['mag'] = 10000.0

	# particle parameters
	params['n_particles'] = args.n_particles

	params['snr'] = 0.05

	if args.snr is not None:
		params['snr'] = args.snr

	# miscellaneous parameters
	params['kernel'] = 'lanczos'
	params['ksize'] = int(6)
	params['rad'] = 0.95
	params['shift_sigma'] = 0
	params['bfactor'] = 50.0

	# Read the volume data and compute fft
	vol,hdr = mrc.readMRC(args.input, inc_header=True)

	params['boxSize'] = int(vol.shape[0])
	params['pxSize'] = (hdr['xlen']/hdr['nx'])

	premult = cryoops.compute_premultiplier(params['boxSize'], params['kernel'], params['ksize']) 

	V = density.real_to_fspace(premult.reshape((1,1,-1)) * premult.reshape((1,-1,1)) * premult.reshape((-1,1,1)) * vol)

	params['sigma_noise'] = vol.std()/params['snr']

	if args.sigma_noise is not None:
		params['sigma_noise'] = args.sigma_noise
		print('Using user-specified sigma_noise.')

	print('Noise Sigma: ' + str(params['sigma_noise']))

	# Set up the particles datastructures
	particles = n.empty((params['n_particles'], params['boxSize'], params['boxSize']), dtype=density.real_t)
	starfile = []
	TtoF = sincint.gentrunctofull(N=params['boxSize'], rad=params['rad'])
	
	tic = time.time()

	results = []

	nChunks = math.ceil(params['n_particles'] / 1000)
	lastChunkSize = params['n_particles'] - ((nChunks - 1)*1000)

	for i in range(nChunks):
		ticc = time.time()
		if i == nChunks - 1:
			chunkSize = lastChunkSize
		else:
			chunkSize = 1000

		# PROCESS IMPLEMENTATION
		manager = mp.Manager()
		output = manager.list()

		jobs = []
		concurrency = mp.cpu_count() - 1

		if args.cpus is not None:
			concurrency = args.cpus

		sema = mp.Semaphore(concurrency)

		print("\nSimulating %d particles on %d processors." % (params['n_particles'], concurrency))

		for j in range(chunkSize):
			idx = i * chunkSize + j
			sema.acquire()
			p = mp.Process(target=simulateParticle, args=(output, params, V, TtoF, idx, tic, sema))
			jobs.append(p)
			p.start()

		for proc in jobs:
			proc.join()

		results.extend(output)

		print("\nDone simulating stack %d of size %d in time %s." % (i, chunkSize, format_timedelta(time.time() - ticc)))

	print("\nDone simulating all particles in: %s" % format_timedelta(time.time() - tic))

	results = sorted(results, key=lambda x: x[0])

	particles = [result[1] for result in results]
	starfile = [result[2] for result in results]

	print('\nWriting out data...')

	# Plot the first 8 images
	fig = plt.figure(figsize=(12, 5))
	col = 4
	row = 2
	for i in range(1, col*row +1):
	    img = particles[i]
	    fig.add_subplot(row, col, i)
	    plt.imshow(img, cmap='gray')
	plt.savefig(args.output_path + 'plot.png')

	mrc.writeMRC(args.output_path + 'simulated_particles.mrcs', n.transpose(particles,(1,2,0)), params['pxSize'])

	# Write the starfile
	f = open((args.output_path + 'simulated_particles.star'), 'w')
	# Write the header
	f.write("\ndata_iparams['mag']es\n\nloop_\n_rlnAmplitudeContrast #1 \n_rlnAnglePsi #2 \n_rlnAngleRot #3 \n_rlnAngleTilt #4 \n_rlnClassNumber #5 \n_rlnDefocusAngle #6 \n_rlnDefocusU #7 \n_rlnDefocusV #8 \n_rlnDetectorPixelSize #9 \n_rlnImageName #10 \n_rlnMagnification #11 \n_rlnOriginX #12 \n_rlnOriginY #13 \n_rlnPhaseShift #14 \n_rlnSphericalAberration #15\n_rlnVoltage #16\n\n")
	# Write the particle information
	for l in starfile:
	    f.write(' '.join(l) + '\n')
	f.close()
	print('Done!')

def simulateParticle(output,params, V, TtoF,i,tic, sema):
	ellapse_time = time.time() - tic
	remain_time = float(params['n_particles'] - i)*ellapse_time/max(i,1)
	print("\r%.2f Percent Complete (%d particles done)... (Elapsed: %s, Remaining: %s)" % (i/float(params['n_particles'])*100.0,i,format_timedelta(ellapse_time),format_timedelta(remain_time)), end="")
		
	# GENERATE PARTICLE ORIENTATION AND CTF PARAMETERS
	p = {}
	# Random orientation vector and get spherical angles
	pt = n.random.randn(3)
	pt /= n.linalg.norm(pt)
	psi = 2*n.pi*n.random.rand()
	
	# Compute Euler angles from a direction vector. Output EA is tuple with phi, theta, psi.
	EA = geom.genEA(pt)[0]
	EA[2] = psi
	
	p['phi'] = EA[0]*180.0/n.pi
	p['theta'] = EA[1]*180.0/n.pi
	p['psi'] = EA[2]*180.0/n.pi
	
	# Compute a random shift
	shift = n.random.randn(2) * params['shift_sigma']
	p['shift_x'] = shift[0]
	p['shift_y'] = shift[1]
	
	# Random defocus within the ranges
	base_defocus = n.random.uniform(params['defocus_min'], params['defocus_max'])
	p['defocus_a'] = base_defocus + n.random.uniform(-500,500)
	p['defocus_b'] = base_defocus + n.random.uniform(-500,500)
	p['astig_angle'] = n.random.uniform(params['defocus_ang_min'], params['defocus_ang_max'])
	
	# CREATE THE PROJECTIONS AND APPLY CTFS
	# Generate rotation matrix based on the Euler Angles
	R = geom.rotmat3D_EA(*EA)[:,0:2]
	slop = cryoops.compute_projection_matrix([R], params['boxSize'], params['kernel'], params['ksize'], params['rad'], 'rots')
	S = cryoops.compute_shift_phases(shift.reshape((1,2)), params['boxSize'], params['rad'])[0]
	D = slop.dot(V.reshape((-1,)))
	D *= S
	  
	# Generate the CTF
	C = ctf.compute_full_ctf(None, params['boxSize'], params['pxSize'], params['accel_kv'], params['spherical_abberr'], params['amp_contrast'], p['defocus_a'], p['defocus_b'], n.radians(p['astig_angle']), 1, params['bfactor'])

	# Apply CTF to the projection and write to particles array
	ctf_distorted = density.fspace_to_real((C*TtoF.dot(D)).reshape((params['boxSize'],params['boxSize'])))
	noise_added = ctf_distorted + n.require(n.random.randn(params['boxSize'], params['boxSize'])*params['sigma_noise'],dtype=density.real_t)
	particle = -noise_added
	
	# Save the particle parameters for the star file
	starfile_line = [str(params['amp_contrast']), 
					 str(p['psi']), 
					 str(p['phi']), 
					 str(p['theta']),
					 str(1),
					 str(p['astig_angle']),                     
					 str(p['defocus_a']),
					 str(p['defocus_b']),
					 str(params['pxSize']),
					 "%d@/simulated_particles.mrcs" % (i+1),
					 str(params['mag']),
					 str(0),
					 str(0),
					 str(0),
					 str(params['spherical_abberr']),
					 str(params['accel_kv'])]

	output.append( (i, particle, starfile_line))
	sema.release()

	# return i, particle, starfile_line

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="input 3d volume", type=str)
	parser.add_argument("output_path", type=str)
	parser.add_argument("--n_particles", help="number of particles to simulate", type=int)
	parser.add_argument("--sigma_noise", help="noise stdev", type=float)
	parser.add_argument("--snr", help="signal to noise ratio", type=float)
	parser.add_argument("--cpus", help="number of processors to use", type=int)


	sys.exit(main(parser.parse_args()))
