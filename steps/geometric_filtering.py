import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from steps.step import Step
import re


from Bio.PDB import PDBIO
from Bio.PDB import PDBParser, Select, PDBIO
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem.Draw import rdMolDraw2D # You'll need this for MolDraw2DCairo/SVG
from rdkit.Chem.Draw.rdMolDraw2D import MolDrawOptions
from rdkit import RDLogger
import tempfile

RDLogger.DisableLog('rdApp.warning')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_pdb_structure(pdb_filepath):
    """
    Loads a PDB structure from a given file path.
    """
    pdb_parser = PDBParser(QUIET=True)
    try:
        structure = pdb_parser.get_structure("prot", pdb_filepath)
        return structure
    except Exception as e:
        raise IOError(f"Error loading PDB structure from {pdb_filepath}: {e}")


def extract_ligand_from_pdb(structure, ligand_smiles, ligand_resname = 'LIG'):
    """
    Extracts a ligand (by residue name) from a Biopython structure and saves it to a new PDB file.
    Loads ligand from a PDB file into RDKit, then assigns bond orders from a provided SMILES string 
    template to ensure correct chemical perception.
    """
    class LigandSelect(Select):
        def accept_residue(self, residue):
            return residue.get_resname().strip() == ligand_resname

    io = PDBIO()
    io.set_structure(structure)

    temp_pdb = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
    io.save(temp_pdb.name, LigandSelect())

    # Load the PDB ligand into RDKit
    pdb_mol = Chem.MolFromPDBFile(temp_pdb.name, removeHs=False)
    if pdb_mol is None:
        raise ValueError("Failed to parse ligand PDB with RDKit.")

    # Create template molecule from SMILES
    template_mol = Chem.MolFromSmiles(ligand_smiles)
    if template_mol is None:
        raise ValueError(f"Could not parse SMILES for template: {ligand_smiles}")

    # Assign bond orders from the template to the PDB-derived molecule
    try:
        ligand_mol = AllChem.AssignBondOrdersFromTemplate(template_mol, pdb_mol)
    except Exception as e:
        print(f"WARNING: Error assigning bond orders from template: {e}")
        print("Proceeding with PDB-parsed molecule (may have incorrect bond orders/valency).")
        ligand_mol = pdb_mol # Fallback if assignment fails

    return ligand_mol


def find_substructure_coordinates(mol, smarts_pattern, atom_to_get_coords_idx=0):
    """
    Finds substructure matches for a given SMARTS pattern in an RDKit molecule
    and returns the 3D coordinates of a specified atom within each match.
    The atom_idx for the carbonyl C and the phosphate atom are 0. 
    """

    coords_dict = {}

    if mol.GetNumConformers() == 0:
        logger.warning("Ligand molecule has no 3D conformers. Cannot get coordinates.")
        return {}

    # Compile SMARTS pattern
    pattern = Chem.MolFromSmarts(smarts_pattern)
    if pattern is None:
        raise ValueError(f"Invalid SMARTS pattern: {smarts_pattern}")
    
    matches = mol.GetSubstructMatches(pattern)
    label = smarts_pattern
    coords_dict[label] = []

    if not matches: 
        logger.warning(f"There was no match found for the SMARTS {smarts_pattern}")
        return coords_dict 

    for match in matches: 
        if atom_to_get_coords_idx >= len(match):
            logger.warning(f"Index {atom_to_get_coords_idx} out of range in match {match}")
            continue

        atom_idx = match[atom_to_get_coords_idx]
        atom = mol.GetAtomWithIdx(atom_idx)
        conf = mol.GetConformer()
        pos = conf.GetAtomPosition(atom_idx)

        coords_dict[label].append({
            'atom': atom.GetSymbol(),
            'coords': (pos.x, pos.y, pos.z)
        })

    return coords_dict


def get_squidly_residue_atom_coords(pdb_path: str, residue_id_str: str):
    '''    
    Extracts squidly residues which are nucleophiles i.e. Ser, Cys or Thr from a PDB file.
    Extracts the 3D coordinates of their atoms and returns them in a dictionary. 
    '''
    # Parse input string into integers
    residue_ids_raw = residue_id_str.split('|')  # This gives you a list of strings
    residue_ids = [int(rid) + 1 for rid in residue_ids_raw] # Because squidly residues are indexed at 0 and PDB files are indexed at 1
    matching_residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM')):
                res_name = line[17:20].strip()             
                res_id = line[22:26].strip()
                atom_name = line[12:16].strip()

                if int(res_id) in residue_ids:

                    if res_name == "SER" and atom_name == "OG":
                        # For SER, we are interested in the OG atom
                        key = f"{res_name}_{res_id}"
                        if key not in matching_residues:
                            matching_residues[key] = []

                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])

                        matching_residues[key].append({
                            'atom': atom_name,
                            'coords': (x, y, z)
                        })

                    elif res_name == "CYS" and atom_name == "SG":
                        # For CYS, we are interested in the SG atom
                        key = f"{res_name}_{res_id}"
                        if key not in matching_residues:
                            matching_residues[key] = []

                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])

                        matching_residues[key].append({
                            'atom': atom_name,
                            'coords': (x, y, z)
                        })

    return matching_residues


def get_all_nucs_atom_coords(pdb_path: str):
    """
    Extracts all nucleophilic residues (Ser, Cys, Thr) from a PDB file.
    Returns a dictionary with residue names as keys and lists of their atom coordinates.
    """
    nucs = ["SER", "CYS", "THR"]
    matching_residues = {}

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(('ATOM')):
                res_name = line[17:20].strip()
                res_id = line[22:26].strip()
                atom_name = line[12:16].strip()

                
                if res_name == "SER" and atom_name == "OG":
                    # For SER, we are interested in the OG atom
                    key = f"{res_name}_{res_id}"
                    if key not in matching_residues:
                        matching_residues[key] = []

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    matching_residues[key].append({
                        'atom': atom_name,
                        'coords': (x, y, z)
                    })
                
                elif res_name == "CYS" and atom_name == "SG":
                    # For CYS, we are interested in the SG atom
                    key = f"{res_name}_{res_id}"
                    if key not in matching_residues:
                        matching_residues[key] = []

                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    matching_residues[key].append({
                        'atom': atom_name,
                        'coords': (x, y, z)
                    })

    return matching_residues


def find_min_distance(ester_dict, squidly_dict): 
    """
    Find the minimum distance between any ester atom (from multiple ester substructures)
    and any nucleophile atom (e.g. from squidly).
    """
    min_dist = float('inf')
    closest_info = None

    for ester_label, ester_atoms in ester_dict.items():
        for ester_atom in ester_atoms:
            coord1 = np.array(ester_atom['coords'])
            lig_atom = ester_atom['atom']

            for nuc_res, nuc_atoms in squidly_dict.items():
                for nuc_atom in nuc_atoms:
                    coord2 = np.array(nuc_atom['coords'])
                    dist = np.linalg.norm(coord1 - coord2)


                    if dist < min_dist:
                        min_dist = dist
                        closest_info = {
                            'ligand_atom': lig_atom,
                            'ligand_substructure': ester_label,
                            'ligand_coords': coord1,  
                            'nuc_res': nuc_res,
                            'nuc_atom': nuc_atom['atom'],
                            'nuc_coords': coord2,     
                            'distance': dist
                        }

    return closest_info


def calculate_dihedral_angle(p1, p2, p3, p4):
    """
    Calculates the dihedral angle between four 3D points.
    Returns the angle in degrees.
    """
    b0 = -1.0 * (np.array(p2) - np.array(p1))
    b1 = np.array(p3) - np.array(p2)
    b2 = np.array(p4) - np.array(p3)

    # Normalize b1 so that it does not influence magnitude of vector
    b1 /= np.linalg.norm(b1)

    # Orthogonal vectors
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))


def calculate_burgi_dunitz_angle(atom_nu_coords, atom_c_coords, atom_o_coords):
    """
    Calculates the Bürgi-Dunitz angle. Defined by the nucleophilic atom (Nu),
    the electrophilic carbonyl carbon (C), and one of the carbonyl oxygen atoms (O).
    """
    # Vectors from carbonyl carbon to nucleophile and to carbonyl oxygen
    vec_c_nu = atom_nu_coords - atom_c_coords
    vec_c_o = atom_o_coords - atom_c_coords

    # Calculate the dot product
    dot_product = np.dot(vec_c_nu, vec_c_o)

    # Calculate the magnitudes of the vectors
    magnitude_c_nu = np.linalg.norm(vec_c_nu)
    magnitude_c_o = np.linalg.norm(vec_c_o)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_c_nu * magnitude_c_o)

    # Ensure cos_angle is within valid range [-1, 1] to prevent arccos errors due to floating point inaccuracies
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


class GeometricFiltering(Step):

    def __init__(self, ligand_smiles: str= '', ester_smarts: str = '', preparedfiles_dir: str = 'filteringpipeline/preparedfiles', output_dir: str= ''):
        
        self.ligand_smiles = ligand_smiles
        self.ester_smarts = ester_smarts
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preparedfiles_dir = Path(preparedfiles_dir)


    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        squidly_residues_col = []
        squidly_distances_col = []
        closest_residue_col = []
        closest_residue_distance = [] 
        burgi_danitz_angle_squidly_res = []
        burgi_danitz_angle_closest_nuc = []

        if not self.preparedfiles_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.preparedfiles_dir}")

        for _, row in df.iterrows():
            entry_name = row['Entry']
            best_structure_name = row['best_structure']
            squidly_residues = row['Squidly_CR_Position']

            try:
                # Load full PDB structure
                pdb_file = self.preparedfiles_dir / f"{best_structure_name}.pdb"
                protein_structure = load_pdb_structure(pdb_file)

                # Extract ligand atoms from PDB
                extracted_ligand_atoms = extract_ligand_from_pdb(protein_structure, self.ligand_smiles )

                # Find phosphate group coordinates
                ester_coords = find_substructure_coordinates(extracted_ligand_atoms, self.ester_smarts, atom_to_get_coords_idx=0) # carbonyl C and phosphate atom are both at index 0

                # Get squidly nucleophile protein atom coordinates
                squidly_atom_coords = get_squidly_residue_atom_coords(pdb_file, squidly_residues)

                # Compute distances between squidly predicted nucleophiles and ester group
                if squidly_atom_coords is None or not squidly_atom_coords:
                    logger.warning(f"No squidly residues found in {entry_name}. ")
                    squidly_residues_col.append(None)
                    squidly_distances_col.append(None)
                    closest_residue_col.append(None)
                    closest_residue_distance.append(None)
                    burgi_danitz_angle_squidly_res.append(None)
                    burgi_danitz_angle_closest_nuc.append(None)

                    continue
                
                squidly_distance = find_min_distance(ester_coords, squidly_atom_coords)

                if squidly_distance:
                    squidly_residues_col.append(squidly_distance['nuc_res'])
                    squidly_distances_col.append(squidly_distance['distance'])

                # Get all nucleophilic residues atom coordinates
                all_nucleophiles_coords = get_all_nucs_atom_coords(pdb_file)

                # Compute smallest distances between all nucleophilic residues and phosphate group
                if all_nucleophiles_coords is None or not all_nucleophiles_coords:
                    logger.warning(f"No nucleophilic residues found in {entry_name}. ")
                    continue

                closest_distance = find_min_distance(ester_coords, all_nucleophiles_coords)
                    
                if closest_distance:
                    closest_residue_col.append(closest_distance['nuc_res'])
                    closest_residue_distance.append(closest_distance['distance'])

                # Calculate Bürgi–Dunitz angle between squidly residue and ester bond respectively closest nucleophile and ester bond
                oxygen_atom_coords = find_substructure_coordinates(extracted_ligand_atoms, self.ester_smarts, atom_to_get_coords_idx=1) # atom1 from SMARTS match (e.g., double bonded O)
                closest_nuc_coords = {
                        closest_distance['nuc_res']: [{
                        'atom': closest_distance['nuc_atom'],
                        'coords': tuple(closest_distance['nuc_coords'])
                    }]
                }
            
                coords_list = []
                coordinates_dictionaries = [squidly_atom_coords, closest_nuc_coords, ester_coords, oxygen_atom_coords]

                for i in coordinates_dictionaries:
                    for atoms in i.values():
                        for atom_info in atoms:
                            coords_list.append(atom_info['coords'])

                coordinates = np.array(coords_list)
                squidly_angle = calculate_burgi_dunitz_angle(coordinates[0], coordinates[2], coordinates[3])
                closest_nuc_angle = calculate_burgi_dunitz_angle(coordinates[1], coordinates[2], coordinates[3])

                burgi_danitz_angle_squidly_res.append(squidly_angle)
                burgi_danitz_angle_closest_nuc.append(closest_nuc_angle)


            except Exception as e:
                logger.error(f"Error processing {entry_name}: {e}")

        return squidly_residues_col, squidly_distances_col, closest_residue_col, closest_residue_distance, burgi_danitz_angle_squidly_res, burgi_danitz_angle_closest_nuc
    

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            squidly_residues_col, squidly_distances_col, closest_residue_col, closest_residue_distance, burgi_danitz_angle_squidly_res, burgi_danitz_angle_closest_nuc = self.__execute(df, self.output_dir)
            df['squidly_residue'] = squidly_residues_col
            df['squidly_residue_ester_distance'] = squidly_distances_col
            df['closest_nuc_residue'] = closest_residue_col
            df['clostest_nuc_residue_distance'] = closest_residue_distance
            df['squidly_is_closest'] = df['squidly_residue'] == df['closest_nuc_residue']
            df['Bürgi–Dunitz angle to squidly residue'] = burgi_danitz_angle_squidly_res
            df['Bürgi–Dunitz angle to closest nucleophile'] = burgi_danitz_angle_closest_nuc
            return df
        else:
            print('No output directory provided')




