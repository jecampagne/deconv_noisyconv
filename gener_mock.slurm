#!/bin/bash
#SBATCH --job-name=gener_mock    # nom du job
#SBATCH -A ixh@v100
#SBATCH -C v100-32g
##SBATCH --qos=qos_gpu-dev  # debug 2h max

#SBATCH --qos=qos_gpu-t4    # 100h maxw


#SBATCH --nodes=1                   # nombre de noeuds
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread        # réservation des cœurs physiques (pas d'hyperthreading)
#SBATCH --gres=gpu:1                # nombre de GPU par nœud (max 8 avec gpu_p2, gpu_p4, gpu_p5)

#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 du noeud ici)
#SBATCH --hint=nomultithread         # hyperthreading desactive

#SBATCH --time=3:00:00              # temps d’execution maximum demande (HH:MM:SS)
#SBATCH --output=gener_mock_%j.out # nom du fichier de sortie
#SBATCH --error=gener_mock_%j.out  # nom du fichier d'erreur (ici commun avec la sortie)
 
# Nettoyage des modules charges en interactif et herites par defaut
module purge
 
 
# Chargement des modules
module load  pytorch-gpu/py3/2.6.0
 
# Echo des commandes lancees
set -x
 
srun python -u ./code/make_dataset.py --file $1
