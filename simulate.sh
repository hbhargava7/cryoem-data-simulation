#!/bin/bash

volnames=("wt" "temet")
volumes=("/run/media/klim/Crucial3/HKB_Vrk1_Synthetic_Maps/6ac9_denmod_bigbox.mrc" "/run/media/klim/Crucial3/HKB_Vrk1_Synthetic_Maps/6ac9TeMet_denmod_bigbox.mrc")

snrs=(200 100 50 5)

for i in "${!volnames[@]}"
do
	for j in "${!snrs[@]}"
	do
		python /run/media/klim/Crucial3/cryoem-data-simulation/simulate_particles.py ${volumes[$i]} "/run/media/klim/Crucial3/simulations/"${volnames[$i]}"_"${snrs[$j]}"/" --cpus 10 --overwrite --sigma_noise ${snrs[$j]} --n_particles 50000
	done
done
