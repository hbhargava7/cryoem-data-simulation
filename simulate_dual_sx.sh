#!/bin/bash

volnames=("wt" "temet")
volumes=("/run/media/klim/Crucial3/HKB_Vrk1_Synthetic_Maps/sxpdb2em_maps/6ac9.mrc" "/run/media/klim/Crucial3/HKB_Vrk1_Synthetic_Maps/sxpdb2em_maps/6ac9TeMet.mrc")

snrs=(1)

for j in "${!snrs[@]}"
do
	python /run/media/klim/Crucial3/cryoem-data-simulation/simulate_dual_stacks.py --input_wt ${volumes[0]} --input_temet ${volumes[1]} --output_path "/run/media/klim/Crucial3/simulations/sx_dual_stack_"${snrs[$j]}"/" --cpus 28 --overwrite --sigma_noise ${snrs[$j]} --n_particles 50000
done
