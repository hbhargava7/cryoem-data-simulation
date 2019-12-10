import sys
import os
import math
import argparse

sys.path.append('cryoem/')
sys.path.append('cryoem/util')

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 100
plt.style.use(['dark_background'])

def main(args):

	theoretical_files = args.theoretical
	experimental_files = args.experimental

	assert len(theoretical_files) == len(experimental_files), 'Error: Please provide the same number of theoretical and experimental data files.'

	theoreticalData = [parse_datafile(i) for i in theoretical_files]
	experimentalData = [parse_datafile(i) for i in experimental_files]

	for i, value in enumerate(zip(theoreticalData, experimentalData)):
		theoretical = value[0]
		experimental = value[1]
		assert len(theoretical) == len(experimental), 'experimental and theoretical starfiles have inconsistent lengths'

		print('read %d particles for pair %i' % (len(experimental), i))


		# Compute angle errors
		print('computing angular errors...')
		angle_errors = computeAngleErrors(theoretical.quaternion, experimental.quaternion)

		angle_mae = mean(angle_errors)

		# Compute position errors
		print('computing positional errors...')
		theoretical_shifts = zip(theoretical.shiftX, theoretical.shiftY)
		experimental_shifts = zip(experimental.shiftX, experimental.shiftY)

		position_errors = computeShiftErrors(theoretical_shifts,experimental_shifts)

		position_mae = mean(position_errors)

		print('plotting...')
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

		# Plot angle errors
		ax1.hist((angle_errors), bins=180, alpha=0.5, color='skyblue', label='wt angle errors');
		ax1.title.set_text('Angle Errors')
		ax1.set_xlabel("Error Distance (°)")
		ax1.set_ylabel("Number of Errors")
		ax1.text(0.1,0.7,"MAE: %.2f" % (angle_mae),transform=ax1.transAxes)
		ax1.legend()

		# Plot position errors
		ax2.hist((position_errors), bins=100, alpha=0.5, color='skyblue', label='position errors');
		ax2.title.set_text('Position Errors')
		ax2.set_xlabel("Error Distance (Å)")
		ax2.set_ylabel("Number of Errors")
		ax2.text(.1,0.7,"MAE: %.2f" % (position_mae),transform=ax2.transAxes)

		ax2.legend()
		plt.savefig('errors.png')
		plt.show()

		print('done!!!')


def parse_datafile(path):
	extension = path[-3:]
	assert extension == 'par' or extension == 'tar', 'Please provide input files as .star or .par files only.'
	if extension == 'par':
		return read_parfile(path)
	else:
		return read_starfile(path)

def read_starfile(path):
	# Read the theoretical starfile
	# We only want (1-indexed): 2 (psi), 3 (phi), 4 (theta), 12 (originX), 13 (originY) 
	# BEWARE skiprows, starfile header lengths may vary
	theoretical = pd.read_csv(path, delim_whitespace=True, header=None, skiprows=21, low_memory=False)
	theoretical = theoretical[theoretical.columns[[1, 2, 3, 11, 12]]]
	theoretical.columns = [ 'psi', 'phi', 'theta', 'shiftX', 'shiftY']
	theoretical = theoretical.astype(float)
	theoretical['quaternion'] = theoretical.apply(lambda row: euler2quat(row.phi*np.pi/180, row.theta*np.pi/180, row.psi*np.pi/180), axis=1)
	return theoretical

def read_parfile(path):
	# Read the experimental parfile
	# BEWARE dropping last two rows.
	experimental = pd.read_csv(path, delim_whitespace=True, low_memory=False)
	experimental = experimental[experimental.columns[[1, 3, 2, 4, 5]]]
	experimental.columns = [ 'psi', 'phi', 'theta', 'shiftX', 'shiftY']
	experimental.drop(experimental.tail(2).index,inplace=True)
	experimental = experimental.astype(float)
	experimental['quaternion'] = experimental.apply(lambda row: euler2quat(row.phi*np.pi/180, row.theta*np.pi/180, row.psi*np.pi/180), axis=1)
	return experimental

def euler2quat(alpha, beta, gamma):
    ha, hb, hg = alpha / 2, beta / 2, gamma / 2
    ha_p_hg = ha + hg
    hg_m_ha = hg - ha
    q = [np.cos(ha_p_hg) * np.cos(hb),
                  np.sin(hg_m_ha) * np.sin(hb),
                  np.cos(hg_m_ha) * np.sin(hb),
                  np.sin(ha_p_hg) * np.cos(hb)]
    return q

# Quaternion to Euler Angles (from https://github.com/asarnow/pyem geom.py)
def quat2euler(q):
    ha1 = np.arctan2(q[1], q[2])
    ha2 = np.arctan2(q[3], q[0])
    alpha = ha2 - ha1  # np.arctan2(r21/r20)
    beta = 2 * np.arccos(np.sqrt(q[0]**2 + q[3]**2))  # np.arccos*r33
    gamma = ha1 + ha2  # np.arctan2(r12/-r02)
    return float(alpha*(180/np.pi)), float(beta*(180/np.pi)), float(gamma*(180/np.pi))

# Angular distance between two quaternions

def quatInverse(q):
    denom = a[0]**2 + a[1]**2 + a[2]**2 + a[3]**2
    return [a[0]/denom, -1*a[1]/denom, -1*a[2]/denom, -1*a[3]/denom]

def quatConj(q):
    # Compute the conjugate of quaternion q
    return [q[0], -1*q[1], -1*q[2], -1*q[3]]

# Angular distance between two quaternions
def quatDist(a,b):
    # Check to verify that quaternions are unit lengths
    assert abs(math.sqrt(a[0]**2+a[1]**2+a[2]**2+a[3]**2)-1)<.001,"a is not a unit quaternion"
    assert abs(math.sqrt(b[0]**2+b[1]**2+b[2]**2+b[3]**2)-1)<.001,"b is not a unit quaternion"
    
    # Compute distance
    s = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
    s = 2*(s**2)-1
    return np.arccos(s)*180/np.pi

# Given two ordered lists of quaternions, compute distances between each angle
def computeAngleErrors(A, B):
	qq = zip(A, B)
	errors = []
	for i,v in enumerate(qq):
		dist = quatDist(v[0],v[1])
		errors.append((dist))
		print(dist)

	return errors

# A and B are (x,y) tuples
def euclideanDistance(A,B):
	return math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

# theoretical and experimental are arrays of (x,y) tuples
def computeShiftErrors(theoretical, experimental):
	ab = zip(theoretical, experimental)
	errors = []
	for i, v in enumerate(ab):
		dist = euclideanDistance(v[0],v[1])
		errors.append(dist)
	return errors

def mean(array):
	return sum(array)/len(array)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--theoretical", help="ordered list of files containing theoretical data", nargs='+', type=str, required=True)
	parser.add_argument("--experimental", help="ordered list of files containing experimental data", nargs='+', type=str, required=True)

	sys.exit(main(parser.parse_args()))