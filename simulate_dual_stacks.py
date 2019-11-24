import sys
import os
import glob
import time
import argparse
import math
import resource
import multiprocessing as mp
import shutil

import pickle

import numpy as n
import numpy.fft as fourier

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
	# Create the output directory
	if not os.path.exists(args.output_path):
		os.mkdir(args.output_path)
	else:
		proceed = False
		if args.overwrite:
			proceed = True
		else:
			proceed = query_yes_no('Output path exists. Overwrite?')
	
		if proceed:
			shutil.rmtree(args.output_path)
			os.mkdir(args.output_path)
		else:
			print('Cancelled.')
			return

	# setup microscope and ctf parameters
	params = {}
	params['defocus_min'] = 10000
	params['defocus_max'] = 20000
	params['defocus_ang_min'] = 0
	params['defocus_ang_max'] = 360
	params['accel_kv'] = 300
	params['amp_contrast'] = 0.07
	params['phase_shift'] = 0
	params['spherical_abberr'] = 2.7
	params['mag'] = 10000.0
	scale = 1

	# particle parameters
	params['n_particles'] = args.n_particles
	
	# miscellaneous parameters
	params['kernel'] = 'lanczos'
	params['ksize'] = int(6)
	params['rad'] = 0.95
	params['shift_sigma'] = 0
	params['bfactor'] = 50.0

	# Read the volume data and compute fft
	print('Volume metadata will be read from the WT volume.')
	vol_wt,hdr_wt = mrc.readMRC(args.input_wt, inc_header=True)
	vol_temet,hdr_temet = mrc.readMRC(args.input_temet, inc_header=True)

	params['boxSize'] = int(vol_wt.shape[0])
	params['pxSize'] = (hdr_wt['xlen']/hdr_wt['nx'])

	premult = cryoops.compute_premultiplier(params['boxSize'], params['kernel'], params['ksize']) 

	V_wt = density.real_to_fspace(premult.reshape((1,1,-1)) * premult.reshape((1,-1,1)) * premult.reshape((-1,1,1)) * vol_wt)
	V_temet = density.real_to_fspace(premult.reshape((1,1,-1)) * premult.reshape((1,-1,1)) * premult.reshape((-1,1,1)) * vol_temet)

	params['wt_signal_mean'] = signalMean(vol_wt)
	params['temet_signal_mean'] = signalMean(vol_temet)

	params['sigma_noise'] = args.sigma_noise

	print('Using specified sigma_noise: ' + str(params['sigma_noise']))

	params['wt_snr'] = params['wt_signal_mean']/params['sigma_noise']
	params['temet_snr'] = params['temet_signal_mean']/params['sigma_noise']

	TtoF = sincint.gentrunctofull(N=params['boxSize'], rad=params['rad'])
	
	tic = time.time()

	nChunks = math.ceil(params['n_particles'] / 1000)
	lastChunkSize = params['n_particles'] - ((nChunks - 1)*1000)

	# Make a directory to cache data on the disk.
	wt_tempPath = args.output_path + 'wt_tmp/'
	if not os.path.exists(wt_tempPath):
		os.mkdir(wt_tempPath)

	temet_tempPath = args.output_path + 'temet_tmp/'
	if not os.path.exists(temet_tempPath):
		os.mkdir(temet_tempPath)

	concurrency = mp.cpu_count() - 1

	if args.cpus is not None:
		concurrency = args.cpus
	print("Simulating %d particles per volume on %d processors." % (params['n_particles'], concurrency))

	for i in range(nChunks):
		ticc = time.time()
		if i == nChunks - 1:
			chunkSize = lastChunkSize
		else:
			chunkSize = 1000

		# PROCESS IMPLEMENTATION
		manager = mp.Manager()
		output_wt = manager.list()
		output_temet = manager.list()

		jobs = []

		sema = mp.Semaphore(concurrency)

		for j in range(chunkSize):
			idx = i * 1000 + j
			sema.acquire()
			p = mp.Process(target=simulateParticles, args=(output_wt, output_temet, params, V_wt, V_temet, TtoF, idx, tic, sema))
			jobs.append(p)
			p.start()

		for proc in jobs:
			proc.join()
			proc.terminate()

		wt_chunkFileName = wt_tempPath + ('%d_chunk.tmp' % i)
		temet_chunkFileName = temet_tempPath + ('%d_chunk.tmp' % i)

		with open(wt_chunkFileName, 'wb') as filehandle:
			pickle.dump(list(output_wt), filehandle)
			filehandle.close()

		with open(temet_chunkFileName, 'wb') as filehandle:
			pickle.dump(list(output_temet), filehandle)
			filehandle.close()

		# print("\nDone simulating chunk %d of size %d in time %s." % (i+1, chunkSize, format_timedelta(time.time() - ticc)))

	print("\nDone simulating all particles in: %s" % format_timedelta(time.time() - tic))
	print("Rate of simulation: %.2f particles PAIRS per second." % (int(params['n_particles'])/float(time.time() - tic)))

	simulation_rate = int(params['n_particles'])/float(time.time() - tic)

	print('Writing out data...')

	particles_wt, starfile_wt = processResultsFromChunkPath(wt_tempPath)
	particles_temet, starfile_temet = processResultsFromChunkPath(temet_tempPath)

	# Plot the first 8 images
	fig = plt.figure(figsize=(12, 5))
	col = 4
	row = 2
	for i in range(1, col*row +1):
		img = particles_wt[i]
		fig.add_subplot(row, col, i)
		plt.imshow(img, cmap='gray')
	plt.savefig(args.output_path + 'wt_plot.png')
	# Plot the first 8 images
	fig = plt.figure(figsize=(12, 5))
	col = 4
	row = 2
	for i in range(1, col*row +1):
		img = particles_temet[i]
		fig.add_subplot(row, col, i)
		plt.imshow(img, cmap='gray')
	plt.savefig(args.output_path + 'temet_plot.png')

	mrc.writeMRC(args.output_path + 'wt_simulated_particles.mrcs', n.transpose(particles_wt,(1,2,0)), params['pxSize'])
	mrc.writeMRC(args.output_path + 'temet_simulated_particles.mrcs', n.transpose(particles_temet,(1,2,0)), params['pxSize'])

	# Write the starfile
	f = open((args.output_path + str(params['sigma_noise']) + '_wt_simulated_particles.star'), 'w')
	# Write the header
	f.write("\ndata_images\n\nloop_\n_rlnAmplitudeContrast #1 \n_rlnAnglePsi #2 \n_rlnAngleRot #3 \n_rlnAngleTilt #4 \n_rlnClassNumber #5 \n_rlnDefocusAngle #6 \n_rlnDefocusU #7 \n_rlnDefocusV #8 \n_rlnDetectorPixelSize #9 \n_rlnImageName #10 \n_rlnMagnification #11 \n_rlnOriginX #12 \n_rlnOriginY #13 \n_rlnPhaseShift #14 \n_rlnSphericalAberration #15\n_rlnVoltage #16\n\n")
	# Write the particle information
	for l in starfile_wt:
		f.write(' '.join(l) + '\n')
	f.close()

	# Write the starfile
	f = open((args.output_path + str(params['sigma_noise']) + '_temet_simulated_particles.star'), 'w')
	# Write the header
	f.write("\ndata_images\n\nloop_\n_rlnAmplitudeContrast #1 \n_rlnAnglePsi #2 \n_rlnAngleRot #3 \n_rlnAngleTilt #4 \n_rlnClassNumber #5 \n_rlnDefocusAngle #6 \n_rlnDefocusU #7 \n_rlnDefocusV #8 \n_rlnDetectorPixelSize #9 \n_rlnImageName #10 \n_rlnMagnification #11 \n_rlnOriginX #12 \n_rlnOriginY #13 \n_rlnPhaseShift #14 \n_rlnSphericalAberration #15\n_rlnVoltage #16\n\n")
	# Write the particle information
	for l in starfile_temet:
		f.write(' '.join(l) + '\n')
	f.close()

	# Write the logfile
	f = open((args.output_path + 'simulation_metadata.txt'), 'w')
	f.write("Thank you for using this data simulator.\n")
	f.write("https://github.com/hbhargava7/cryoem-data-simulation\n\n")
	f.write("Simulated %d particle pairs in %s.\n" % (params['n_particles'], format_timedelta(time.time() - tic)))
	f.write("Rate of simulation was %.2f particle PAIRS per second." % simulation_rate)
	f.write("\n\nInput wt volume: %s.\n" % args.input_wt)
	f.write("\n\nInput temet volume: %s.\n" % args.input_temet)

	f.write("Output path: %s.\n\n" % args.output_path)

	if args.sigma_noise is not None:
		f.write("Used user-specified noise sigma: " + str(params['sigma_noise']))
	else:
		f.write("Used snr-based noise sigma: " + str(params['sigma_noise']))

	params_string = "{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in params.items()) + "}"
	f.write("\n\n\nParameters Dump: \n" + str(params_string))

	f.close()

	print('Done!')

def processResultsFromChunkPath(tempPath):
	results = []
	chunkFiles = [f for f in os.listdir(tempPath) if os.path.isfile(os.path.join(tempPath, f))]
	tempPath = os.path.abspath(tempPath)

	for f in chunkFiles:
		file = open(os.path.join(tempPath, f), 'rb')
		chunk = pickle.load(file)
		results.extend(chunk)

	# Delete the temp directory
	shutil.rmtree(tempPath)

	results = sorted(results, key=lambda x: x[0])

	particles = [result[1] for result in results]
	starfile = [result[2] for result in results]

	return particles,starfile
def signalMean(volume):
	# Compute the mean of the signal, excluding zeros

	nonzero = volume
	nonzero[nonzero == 0] = n.nan
	return n.nanmean(nonzero)

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def simulateParticles(output_wt, output_temet, params, V_wt, V_temet, TtoF, i, tic, sema):
	ellapse_time = time.time() - tic
	remain_time = float(params['n_particles'] - i)*ellapse_time/max(i,1)
	print("\r%.2f Percent Complete (%d particle pairs done)... (Elapsed: %s, Remaining: %s)" % ((i+1)/float(params['n_particles'])*100.0,i+1,format_timedelta(ellapse_time),format_timedelta(remain_time)), end="")
		
	# Numpy random seed
	n.random.seed(int.from_bytes(os.urandom(4), byteorder='little')) 

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


	D_wt = slop.dot(V_wt.reshape((-1,)))
	D_wt *= S

	D_temet = slop.dot(V_temet.reshape((-1,)))
	D_temet *= S
	  
	# Generate the CTF
	C = ctf.compute_full_ctf(None, params['boxSize'], params['pxSize'], params['accel_kv'], params['spherical_abberr'], params['amp_contrast'], p['defocus_a'], p['defocus_b'], n.radians(p['astig_angle']), 1, params['bfactor'])

	# Apply CTF to the projection and write to particles array
	wt_ctf_distorted = density.fspace_to_real((C*TtoF.dot(D_wt)).reshape((params['boxSize'],params['boxSize'])))
	temet_ctf_distorted = density.fspace_to_real((C*TtoF.dot(D_temet)).reshape((params['boxSize'],params['boxSize'])))

	noise = n.require(n.random.randn(params['boxSize'], params['boxSize'])*params['sigma_noise'],dtype=density.real_t)

	wt_noise_added = wt_ctf_distorted + noise
	temet_noise_added = temet_ctf_distorted + noise

	wt_particle = -wt_noise_added
	temet_particle = -temet_noise_added

	# Save the particle parameters for the star file
	wt_starfile_line = [str(params['amp_contrast']), 
					 str(p['psi']), 
					 str(p['phi']), 
					 str(p['theta']),
					 str(1),
					 str(p['astig_angle']),                     
					 str(p['defocus_a']),
					 str(p['defocus_b']),
					 str(params['pxSize']),
					 "%d@/%s_wt_simulated_particles.mrcs" % (i+1, str(params['sigma_noise'])),
					 str(params['mag']),
					 str(0),
					 str(0),
					 str(0),
					 str(params['spherical_abberr']),
					 str(params['accel_kv'])]

	temet_starfile_line = [str(params['amp_contrast']), 
				 str(p['psi']), 
				 str(p['phi']), 
				 str(p['theta']),
				 str(1),
				 str(p['astig_angle']),                     
				 str(p['defocus_a']),
				 str(p['defocus_b']),
				 str(params['pxSize']),
				 "%d@/%s_temet_simulated_particles.mrcs" % (i+1,str(params['sigma_noise'])),
				 str(params['mag']),
				 str(0),
				 str(0),
				 str(0),
				 str(params['spherical_abberr']),
				 str(params['accel_kv'])]

	output_wt.append((i, wt_particle, wt_starfile_line))
	output_temet.append((i, temet_particle, temet_starfile_line))

	sema.release()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input_wt", help="input wild-type 3d volume", type=str, required=True)
	parser.add_argument("--input_temet", help="input telluromethionine-type 3d volume", type=str, required=True)
	parser.add_argument("--output_path", help="output path",type=str, required=True)
	parser.add_argument("--n_particles", help="number of particles to simulate", type=int, required=True)
	parser.add_argument("--sigma_noise", help="noise stdev", type=float, required=True)
	parser.add_argument("--cpus", help="number of processors to use", type=int)
	parser.add_argument("--overwrite", help="overwrite the target directory if necessary?", action='store_true')

	sys.exit(main(parser.parse_args()))
