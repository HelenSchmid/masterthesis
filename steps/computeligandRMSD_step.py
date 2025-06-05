import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from step import Step

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from spyrmsd import rmsd
from pathlib import Path
from scipy.spatial.distance import cdist  

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


element_to_atomic_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
    "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28,
    "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
    "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46,
    "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65,
    "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83,
    "Th": 90, "Pa": 91, "U": 92
}

def extract_two_ligands_from_pdb(pdb_path):
    with open(pdb_path, 'r') as f:
        pdb_file = pdb.PDBFile.read(f)
    structure = pdb.get_structure(pdb_file, model=1)

    is_ligand = ~struc.filter_amino_acids(structure) & ~struc.filter_solvent(structure)
    ligands = structure[is_ligand]
    
    ligand_list = []
    for chain_id in set(ligands.chain_id):
        chain_lig = ligands[ligands.chain_id == chain_id]
        ligand_list.append(chain_lig)

    if len(ligand_list) != 2:
        raise ValueError("Expected exactly 2 ligands, found: {}".format(len(ligand_list)))
    return ligand_list[0], ligand_list[1]

# Function to calculate adjacency matrix from coordinates and atomic numbers
def calculate_adjacency_matrix(coords, threshold=2.0):
    # Calculate pairwise distances between atoms
    distances = cdist(coords, coords)
    
    # Create the adjacency matrix where distances below threshold indicate bonds
    adjacency_matrix = (distances < threshold) & (distances > 0)  # Avoid self-bonds
    return adjacency_matrix.astype(int)  # Convert to integer (0 or 1)


def prepare_for_symmrmsd(ligand):
    coords = struc.coord(ligand)
    atomic_nums = np.array([element_to_atomic_number[el] for el in ligand.element])
    adjacency = calculate_adjacency_matrix(coords)  # Create the adjacency matrix
    return coords, atomic_nums, adjacency


class LigandRMSD(Step):
    def __init__(self, pdb_file = str, output_dir =  str, num_threads=1): 
        self.pdb_file = Path('/home/helen/cec_degrader/generalize/FilteringPipeline/ScreenedVariants/Biotite/P96084')    
        self.output_dir = output_dir or None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        rmsd_values = []

        #for pdb_file in self.pdb_file: 

            # Check if the directory exists
            #if not pdb_file.exists() or not pdb_file.is_dir():
            #   raise ValueError(f"The provided path {pdb_file} is not a valid directory.")

        # Get all PDB files in the directory
        pdb_files = list(self.pdb_file.glob("*.pdb"))  # Use glob to match all .pdb files
        
        for pdb_file_path in pdb_files: 

            ligand_a, ligand_b = extract_two_ligands_from_pdb(pdb_file_path)
            coords_a, atomic_nums_a, adj_a = prepare_for_symmrmsd(ligand_a)
            coords_b, atomic_nums_b, adj_b = prepare_for_symmrmsd(ligand_b)

            symm_rmsd_value = rmsd.symmrmsd(
                coordsref=coords_a,                 # Reference coordinated
                coords=coords_b,                    # Coordinates (one set or multiple sets)
                apropsref=atomic_nums_a,            # Reference atomic properties
                aprops=atomic_nums_b,               # Atomic properties
                amref=adj_a,                        # Reference adjacency matrix
                am=adj_b,                           # Adjacency matrix
                center=False,                       # Flag to center molecules at origin
                minimize=False                      # Flag to compute minimum RMSD
            )

            # Store the RMSD value in a dictionary to append later
            pdb_file_name = pdb_file_path.name
            ligand_names = pdb_file_name.replace(".pdb", "").split("__")

            docked_structure1_name = ligand_names[0] if len(ligand_names) > 0 else None
            docked_structure2_name = ligand_names[1] if len(ligand_names) > 1 else None

            rmsd_values.append({
                'Entry': 'P96084', 
                'pdb_file': pdb_file_path.name,  # Store the name of the PDB file
                'docked_structure1' : docked_structure1_name, 
                'docked_structure2' : docked_structure2_name, 
                'ligand_rmsd': symm_rmsd_value  # Store the calculated RMSD value
            })

        # Convert the list of dictionaries into a DataFrame
        rmsd_df = pd.DataFrame(rmsd_values)

        # Save the DataFrame as a pickle file
        csv_file = tmp_dir / "ligand_rmsd.csv"
        rmsd_df.to_csv(csv_file)
        logger.info(f"Ligand RMSD results saved to: {csv_file}")

        return csv_file

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames += self.__execute(df_chunk, self.output_dir)
                    
                df['ligandRMSD_dir'] = output_filenames
                return df
            
            else:
                output_filenames = self.__execute(df, self.output_dir)
                df['ligandRMSD_dir'] = output_filenames
                return df
        else:
            print('No output directory provided')

