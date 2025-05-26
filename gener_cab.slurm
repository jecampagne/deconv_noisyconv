#!/bin/bash
#SBATCH --job-name=gener_cab    # nom du job
#SBATCH -A ixh@v100

#SBATCH -C v100-32g
##SBATCH --qos=qos_gpu-dev  # debug 2h max

#SBATCH --partition=prepost
#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)
##SBATCH --gres=gpu:1                # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)


#SBATCH --time=1:30:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=gener_cab_%j.out # nom du fichier de sortie
#SBATCH --error=gener_cab_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
 
# Chargement des modules
 module load  pytorch-gpu/py3/2.6.0
 
# Echo des commandes lancees
set -x
 
srun python -u ./code/make_Calphabeta_image.py --file $1
