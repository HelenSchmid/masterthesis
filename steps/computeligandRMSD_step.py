from steps.step import Step


import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from biotite.structure.io.pdb import PDBFile
from spyrmsd import rmsd
from scipy.spatial.distance import cdist  
from openbabel import openbabel as ob
from openbabel import pybel
import tempfile

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

def write_ligand_to_temp_pdb(ligand):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(ligand)
    with open(tmp.name, "w") as f:
        pdb_file.write(f)
    return tmp.name

def get_obabel_ligand_properties(pdb_path):
    mol = next(pybel.readfile("pdb", str(pdb_path)))
    mol.addh()  # Ensure consistent hydrogen handling

    atoms = list(mol.atoms)
    N = len(atoms)

    coords = np.array([atom.coords for atom in atoms])
    atomic_nums = np.array([atom.atomicnum for atom in atoms])
    adjacency = np.zeros((N, N), dtype=int)

    for bond in ob.OBMolBondIter(mol.OBMol):
        i = bond.GetBeginAtomIdx() - 1
        j = bond.GetEndAtomIdx() - 1
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    return coords, atomic_nums, adjacency

def visualize_rmsd_by_entry(rmsd_df, output_dir="ligandRMSD_heatmaps"):
    '''
    Visualizes RMSD values as heatmaps for each entry in the resulting dataframe.
    '''   
    os.makedirs(output_dir, exist_ok=True)

    for entry, group in rmsd_df.groupby('Entry'):
        # Get all docked structures for the entry
        docked_proteins = list(set(group['docked_structure1']) | set(group['docked_structure2']))
        docked_proteins = sorted(docked_proteins, key=lambda x: (0 if "chai" in x.lower() else 1, x))
    
        rmsd_matrix = pd.DataFrame(np.nan, index=docked_proteins, columns=docked_proteins)

        for _, row in group.iterrows():
            l1, l2, rmsd = row['docked_structure1'], row['docked_structure2'], row['ligand_rmsd']
            rmsd_matrix.loc[l1, l2] = rmsd
            rmsd_matrix.loc[l2, l1] = rmsd


        plt.figure(figsize=(6, 5))
        sns.heatmap(rmsd_matrix, annot=False, cmap="viridis", square=True, cbar=True)
        plt.title(f"RMSD Heatmap: {entry}", fontsize=14)
        plt.xlabel("Docked Structures")
        plt.ylabel("Docked Structures")
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        filename = f"{entry.replace('/', '_')}_heatmap.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def select_best_docked_structures(rmsd_df: pd.DataFrame, output_dir: Path | str = None) -> pd.DataFrame:
    """
    For each Entry, selects the docked structure with the lowest average RMSD to all others.
    """
    best_structures = []

    for entry, group in rmsd_df.groupby("Entry"):
        # Unique structure names
        structures = list(set(group['docked_structure1']).union(group['docked_structure2']))
        structures.sort()

        # Create empty symmetric matrix
        rmsd_matrix = pd.DataFrame(np.nan, index=structures, columns=structures)

        for _, row in group.iterrows():
            s1, s2, r = row['docked_structure1'], row['docked_structure2'], row['ligand_rmsd']
            rmsd_matrix.loc[s1, s2] = r
            rmsd_matrix.loc[s2, s1] = r

        # Fill diagonal with 0
        np.fill_diagonal(rmsd_matrix.values, 0)

        # Calculate mean RMSD for each structure
        avg_rmsd = rmsd_matrix.mean(axis=1)

        best_structure = avg_rmsd.idxmin()
        squidly_residues = rmsd_df.loc[rmsd_df['Entry'] == entry, 'Squidly_CR_Position']

        best_structures.append({
            'Entry': entry,
            #'pdb_file': rmsd_df.loc[rmsd_df['Entry'] == entry, 'pdb_file'], 
            'best_structure': best_structure,
            'avg_rmsd': avg_rmsd[best_structure],
            'Squidly_CR_Position': squidly_residues.iloc[0] if not squidly_residues.empty else None
        })

    best_df = pd.DataFrame(best_structures)

    if output_dir:
        output_path = Path(output_dir) / "best_docked_structures.csv"
        best_df.to_csv(output_path, index=False)
        logger.info(f"Best docked structures saved to: {output_path}")

    return best_df



class LigandRMSD(Step):
    def __init__(self, entry_col = 'Entry', input_dir: str = '', output_dir: str = '', visualize_heatmaps = False,  num_threads=1): 
        self.entry_col = entry_col
        self.input_dir = Path(input_dir)   
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualize_heatmaps = visualize_heatmaps
        self.num_threads = num_threads or 1

    def __execute(self, df) -> list:

        rmsd_values = []

        # Iterate through all subdirectories in the input directory
        for sub_dir in self.input_dir.iterdir():
            print(f"Processing subdirectory: {sub_dir}")

            # Process all PDB files in subdirectories
            for pdb_file_path in sub_dir.glob("*.pdb"):

                ligand1, ligand2 = extract_two_ligands_from_pdb(pdb_file_path)

                ligand1_path = write_ligand_to_temp_pdb(ligand1)
                ligand2_path = write_ligand_to_temp_pdb(ligand2)

                coords1, atomic_nums1, adj1 = get_obabel_ligand_properties(ligand1_path)
                coords2, atomic_nums2, adj2 = get_obabel_ligand_properties(ligand2_path)

                try:
                    if coords1.shape != coords2.shape:
                        raise ValueError("Mismatched ligand atom count.")
                    
                    symm_rmsd_value = rmsd.symmrmsd(
                        coordsref=coords1,                 # Reference coordinated
                        coords=coords2,                    # Coordinates (one set or multiple sets)
                        apropsref=atomic_nums1,            # Reference atomic properties
                        aprops=atomic_nums2,               # Atomic properties
                        amref=adj1,                        # Reference adjacency matrix
                        am=adj2,                           # Adjacency matrix
                        center=False,                      # Flag to center molecules at origin
                        minimize=False                     # Flag to compute minimum RMSD
                    )
                except Exception as e:
                    print(f"Failed RMSD calculation for pair: {ligand1_path}, {ligand2_path} — {e}")
                    continue

                # Store the RMSD value in a dictionary to append later
                pdb_file_name = pdb_file_path.name
                structure_names = pdb_file_name.replace(".pdb", "").split("__")
                
                docked_structure1_name = structure_names[0] if len(structure_names) > 0 else None
                docked_structure2_name = structure_names[1] if len(structure_names) > 1 else None

                entry_name = docked_structure1_name.split('_')[0]
                squidly_residues = df.loc[df[self.entry_col] == entry_name.strip(), 'Squidly_CR_Position']

                rmsd_values.append({
                    'Entry': entry_name, 
                    'pdb_file': pdb_file_path.name,  # Store the name of the PDB file
                    'docked_structure1' : docked_structure1_name, 
                    'docked_structure2' : docked_structure2_name, 
                    'ligand_rmsd': symm_rmsd_value,   # Store the calculated RMSD value
                    'Squidly_CR_Position': squidly_residues.iloc[0] if not squidly_residues.empty else None
                })

        # Convert the list of dictionaries into a DataFrame
        rmsd_df = pd.DataFrame(rmsd_values)


        # If heatmaps are to be visualized, call the visualization function
        if self.visualize_heatmaps:
            heatmap_output_dir = Path(self.output_dir) / 'ligandRMSD_heatmaps'
            os.makedirs(heatmap_output_dir, exist_ok=True)
            visualize_rmsd_by_entry(rmsd_df, output_dir=heatmap_output_dir)

        # Select the best docked structures based on RMSD
        select_best_docked_structures(rmsd_df, output_dir=self.output_dir)

        # Save the DataFrame as a csv file
        csv_file = Path(self.output_dir) / "ligand_rmsd.csv"
        rmsd_df.to_csv(csv_file, index=False)
        logger.info(f"Ligand RMSD results saved to: {csv_file}")
        return rmsd_df


    def execute(self, df) -> pd.DataFrame:
        self.input_dir = Path(self.input_dir)
        return self.__execute(df)
