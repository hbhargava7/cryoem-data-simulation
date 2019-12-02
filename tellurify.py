# tellurify.py -i 6ac9.pdb -o 6ac9_TePhe.pdb --type tephe --exclude_residues 2 5 --ringflip_residues 3 5

# Note: The TePhe ring plane is aligned with the existing Phe ring plane using a brute force method. This can be substantially improved (see notebook).

import sys
import copy
import math
import argparse

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from Bio.PDB import *

def coordsOfAtomFromResidue(atomName, resi):
	return [a for a in resi.get_atoms() if a.get_name() == atomName][0].get_coord()

def quatConj(q):
	# Compute the conjugate of quaternion q
	return [q[0], -1*q[1], -1*q[2], -1*q[3]]

def quatMult(q1, q2):
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
	z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
	return w, x, y, z

def quatToRotateAontoB(a,b):
	theta = np.arccos(np.dot(a,b))
	axis = normalize(np.cross(a, b))
	s = np.sin(theta/2)
	q = [np.cos(theta/2), axis[0]*s, axis[1]*s, axis[2]*s]
	q = q / np.linalg.norm(q)
	return q

def transformVectorByQuaternion(v, q):
	# Transform a 3d vector by quaternion q, and return a 3D vector
	vq = [0, v[0], v[1], v[2]]
	transformed = quatMult(quatMult(q, vq), quatConj(q))
	
	return [transformed[1], transformed[2], transformed[3]]
	
def normalize(v):
	norm = np.linalg.norm(v)
	return v / norm

def bruteForceAlignRingOrientation(tephe0, phe, offset=0):
	 # Get plane normals by crossproduct of two vectors in each ring
	pGD1 = normalize(np.subtract(phe['CD1'], phe['CG']))
	pGD2 = normalize(np.subtract(phe['CD2'], phe['CG']))
	tGD = normalize(np.subtract(tephe0['CD'], tephe0['CG']))
	tGTe = normalize(np.subtract(tephe0['Te'], tephe0['CG']))
	
	tBG = normalize(np.subtract(tephe0['CB'],tephe0['CG']))
	tGZ = normalize(np.subtract(tephe0['CZ'], tephe0['CG']))
	pGZ = normalize(np.subtract(phe['CZ'], phe['CG']))
	phe_norm = normalize(np.cross(pGD1, pGD2))
	tephe_norm = normalize(np.cross(tGD, tGTe))
	
	ring_delta = np.arccos(np.dot(tephe_norm, phe_norm))    
	
	thetas = []
	deltas = []
	tephes = []

	for theta in np.linspace(0, 360, num=3600):
		tephe = copy.deepcopy(tephe0)
		
		# Generate the rotation
		theta = theta*np.pi/180
		offset = offset*np.pi/180

		axis = normalize(pGZ)
		s = np.sin(theta/2)
		q = [np.cos(theta/2), axis[0]*s, axis[1]*s, axis[2]*s]
		q = q / np.linalg.norm(q)

		# Apply the quaternion rotation to the tephe points
		for key,value in tephe.items():
			tephe[key] = transformVectorByQuaternion(value, q)
		
		## MEASURE THE RING ANGLE AGAIN
		pGD1 = normalize(np.subtract(phe['CD1'], phe['CG']))
		pGD2 = normalize(np.subtract(phe['CD2'], phe['CG']))

		tGD = normalize(np.subtract(tephe['CD'], tephe['CG']))
		tGTe = normalize(np.subtract(tephe['Te'], tephe['CG']))

		tBG = normalize(np.subtract(tephe['CG'],tephe['CB']))

		phe_norm = normalize(np.cross(pGD1, pGD2))
		tephe_norm = normalize(np.cross(tGD, tGTe))

		ring_delta = np.arccos(np.dot(tephe_norm, phe_norm))
		
		thetas.append(theta) # input values (how much the ring was rotated)
		deltas.append(ring_delta) # orientation difference between rings
		tephes.append(tephe) # the actual coordinates

	deltas = np.asarray(deltas)
	idx = (np.abs(deltas - offset)).argmin()
	
	return tephes[idx], deltas[idx]

def main(args):

	# if args.exclude_residues == None:
	# 	args.exclude_residues = []
	# if args.ringflip_residues == None:

	# Import data from the PDB
	parser = PDBParser()
	structure = parser.get_structure('target', args.input)

	if args.type == 'temet':
		M = [r for r in structure[0]['A'] if r.get_resname() == "MET"]

		print("Loaded PDB and found %d methionines." % len(M))

		for m in M:
			if (m.get_id()[1]) in args.exclude_residues:
				continue

			m.resname = 'TMT'
			
			# Remove the PHE atoms after CA from the PDB file. Also set the position of CB.
			toDelete = []

			tellurium = None
			toDelete = []
			for atom in m.get_atoms():
				if atom.get_name() == 'SD':
					toDelete.append(atom.get_id())
					tellurium = Atom.Atom('TE', atom.get_coord(), atom.get_bfactor(), atom.get_occupancy(), atom.get_altloc(), 'Te', 'TE', 'TE')

			for i in toDelete:
				m.detach_child(i)

			m.add(tellurium)
		print("Substitution done, writing TeMet coordinates.")

	elif args.type == 'tephe':

		F = [r for r in structure[0]['A'] if r.get_resname() == "PHE"]

		print("Loaded PDB and found %d phenylalanines." % len(F))

		# Get the TePhe rigid body
		tephe_structure = parser.get_structure('tephe', 'reference/tephe_rectified.pdb')
		tephe_resi = list(tephe_structure[0].get_residues())[0]

		# Make a dictionary with all the necessary TePhe atoms (ring plus beta carbon)
		tephe0 = {
			'CB' : coordsOfAtomFromResidue('CB',tephe_resi),
			'CG' : coordsOfAtomFromResidue('CG',tephe_resi),
			'CD' : coordsOfAtomFromResidue('CD',tephe_resi),
			'CE' : coordsOfAtomFromResidue('CE',tephe_resi),
			'CZ' : coordsOfAtomFromResidue('CZ',tephe_resi),
			'Te' : coordsOfAtomFromResidue('Te',tephe_resi)
		}

		# Iterate through the phenylalanines
		for i, f in enumerate(F):
			if f.get_id()[1] in args.exclude_residues:
				continue			

			tephe = copy.deepcopy(tephe0)
			# Get the PHE atomic coordinates needed to position the new TEPHE ring.
			phe = {
				'CA' : coordsOfAtomFromResidue('CA',f),
				'CB' : coordsOfAtomFromResidue('CB',f),
				'CG' : coordsOfAtomFromResidue('CG',f),
				'CD1' : coordsOfAtomFromResidue('CD1',f),
				'CD2' : coordsOfAtomFromResidue('CD2',f),
				'CE1' : coordsOfAtomFromResidue('CE1',f),
				'CE2' : coordsOfAtomFromResidue('CE2',f),
				'CZ' : coordsOfAtomFromResidue('CZ',f)
			}
						
			# STEP 1: ALIGN THE STEM VECTORS
			# Find the stem vectors between beta and gamma carbons
			tBG = normalize(np.subtract(tephe['CG'],tephe['CB']))
			pBG = normalize(np.subtract(phe['CG'],phe['CB']))
			
			# Calculate the quaternion rotation that makes tBG parallel to pBG
			q = quatToRotateAontoB(tBG, pBG)

			# Apply the quaternion rotation to the tephe points
			for key,value in tephe.items():
				tephe[key] = transformVectorByQuaternion(value, q)
			
			# STEP 2: ALIGN THE RING PLANES
			# Using a brute force approach. It is unclear why the principled approach wasn't working.
			offset = 0
			if f.get_id()[1] in args.ringflip_residues or args.ringflip_all:
				offset = 180
			tephe, ring_delta = bruteForceAlignRingOrientation(tephe, phe, offset)
			
			print("Ring Delta for %d: %0.2f" % (i,ring_delta*(180/np.pi)))
				
			# Translate the ring so that tCB aligns with pCB
			translation = np.subtract(phe['CB'],tephe['CB'])
			for key,value in tephe.items():
				tephe[key] = np.add(value, translation)
				
			# Construct the TePhe residue (N, CA, C, O, CB, CG, CD, CE, CZ, Te)
			f.resname = 'TPE'
			
			# Remove the PHE atoms after CA from the PDB file. Also set the position of CB.
			toDelete = []
			for atom in f.get_atoms():
				if atom.get_name() in ['CD1', 'CD2', 'CE1', 'CE2', 'CZ']:
					toDelete.append(atom.get_id())
				elif atom.get_name() == 'CB':
					atom.set_coord(tephe['CB'])
				elif atom.get_name() == 'CG':
					atom.set_coord(tephe['CG'])

			for i in toDelete:
				f.detach_child(i)
		#               name, coord, bf, occ, altloc, fullname, serial, element
			cd = Atom.Atom('CD', tephe['CD'], 0, 1, ' ', 'CD', 'C', 'C')
			ce = Atom.Atom('CE', tephe['CE'], 0, 1, ' ', 'CE', 'C', 'C')
			cz = Atom.Atom('CZ', tephe['CZ'], 0, 1, ' ', 'CZ', 'C', 'C')
			te = Atom.Atom('Te', tephe['Te'], 0, 1, ' ', 'Te', 'C', 'TE')

			f.add(cd)
			f.add(ce)
			f.add(cz)
			f.add(te)
		print("Substitution done, writing TePhe coordinates.")

					
	io = PDBIO()
	io.set_structure(structure)
	io.save(args.output)
	print("Done!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', help="input PDB file", type=str, required=True)
	parser.add_argument('-o','--output', help="output path", type=str, required=True)
	parser.add_argument('--type', help="type (tephe or temet)", type=str, required=True)

	parser.add_argument('--ringflip_all', help="ringflip all residues", action='store_true')
	parser.add_argument('--exclude_residues', help="indexes of residues to exclude (separate with spaces)", nargs='+', type=int, required=False, default=[])
	parser.add_argument('--ringflip_residues', help="indexes of residues to ringflip (separate with spaces)", nargs='+', type=int, required=False, default=[])

	sys.exit(main(parser.parse_args()))
