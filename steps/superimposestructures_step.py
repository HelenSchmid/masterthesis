import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from step import Step

import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io.pdb as pdb

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_structure(type, file):
    if type == 'PDB':
        structure = pdb.get_structure(file, model=1)
    if type == 'PDBx':
        structure = pdbx.get_structure(file, model=1)
    return structure

def extract_ligand(structure):
    # Apply filters: select non-amino acids and non-solvents (ligands)
    is_ligand = ~struc.filter_amino_acids(structure) & ~struc.filter_solvent(structure)
    
    # Return only the ligand atoms from the structure
    return structure[is_ligand]

def extract_monomer(complex):
    complex = complex[struc.filter_amino_acids(complex)]
    # Get the monomer that belongs to the first atom in the structure
    return complex[struc.get_chain_masks(complex, [0])[0]]

import re
from pathlib import Path


def clean_vina_ligand_file(input_path, output_path=None):
    """
    Cleans a Vina ligand PDBQT file by:
    - Removing the first line and the line immediately following any 'ENDMDL'.
    - Relabeling ATOM records to HETATM.
    - Appending a number to the atom names (e.g., C1, C2, O1, O2) per atom type.
    - Saves the cleaned content to a new file.
    
    Parameters:
    input_path (str or Path): The path to the original file.
    output_path (str or Path, optional): Where to save the cleaned file. 
                                         If not provided, '_cleaned' is appended to the filename.
    
    Returns:
    Path: Path to the cleaned output file.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")

    with open(input_path, 'r') as file:
        lines = file.readlines()

    filtered_lines = []
    skip_next_line = False
    atom_counter = {'C': 1, 'O': 1, 'P': 1, 'N': 1}  # Separate counters for each atom type

    for i, line in enumerate(lines[1:], start=1):  # Skip the first line
        if skip_next_line:
            skip_next_line = False
            continue
        if line.startswith('ENDMDL'):
            filtered_lines.append(line)
            skip_next_line = True
        else:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                # Extract the residue and atom name from the line
                parts = list(line.strip())
                residue_name = ''.join(parts[17:20]).strip()
                atom_name = ''.join(parts[12:16]).strip()

                # Atom type (first letter of the atom name)
                atom_type = atom_name[0]

                # Update the atom name by appending the counter number
                if atom_type in atom_counter:
                    parts[12:16] = f'{atom_type}{atom_counter[atom_type]}'.ljust(4)  # Append number to atom name
                    atom_counter[atom_type] += 1
                else:
                    # If atom type is not tracked, initialize the counter
                    atom_counter[atom_type] = 1
                    parts[12:16] = f'{atom_type}{atom_counter[atom_type]}'.ljust(4)
                    atom_counter[atom_type] += 1

                # Change ATOM to HETATM
                parts[0:6] = 'HETATM'

                # Reassemble the line with the new atom name
                new_line = ''.join(parts)
                filtered_lines.append(new_line + '\n')
            else:
                filtered_lines.append(line)

    with open(output_path, 'w') as file:
        file.writelines(filtered_lines)

    return output_path


def clean_pdb_atom_names(pdb_file):
    """
    Removes suffixes like '_1', '_2' from atom names in a Biotite PDBFile object.

    Parameters:
    pdb_file (PDBFile): A Biotite PDBFile object with structure data.

    Returns:
    PDBFile: Modified PDBFile with cleaned atom names.
    """
    atom_array = pdb_file.get_structure()
    cleaned_names = [
        re.sub(r'(_\d+|_)$', '', name)  # Removes "_", "_1", "_23", etc.
        for name in atom_array.atom_name
    ]
    atom_array.atom_name = cleaned_names
    pdb_file.set_structure(atom_array)
    return pdb_file


def truncate_atom_names(structure):
    # Truncate atom names to the first 4 characters
    structure.atom_name = [atom[:4] for atom in structure.atom_name]
    return structure

def truncate_residue_names(structure):
    structure.res_name = [res[:3] for res in structure.res_name]  # Truncate to 3 characters
    return structure

def get_residue_ids(structure):
    """Returns a set of (res_id, ins_code) for each residue."""
    return set(zip(structure.res_id, structure.ins_code))


class SuperimposeStructures(Step):
    def __init__(self, chai_col = None, vina_col = None, placer_col = None,  ligand_name: str = "", output_dir: str = '', num_threads=1): #placer_col=None, AF3_col=None,
        self.chai_col = chai_col
        self.vina_col = vina_col
        self.placer_col = placer_col
        #self.AF3_col = AF3_col
        self.ligand_name = ligand_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

            ## Ensure output directory exists
            #output_dir = self.output_dir / "superimposed3"
            #output_dir.mkdir(parents=True, exist_ok=True)

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        # Validate that at least 2 structures are provided
        self.structure_cols = [col for col in [self.vina_col, self.chai_col, self.placer_col] if col]
        if len(self.structure_cols) < 2:
            raise ValueError("At least two structure columns must be provided for superposition.")

        chai_structures = {}
        chai_ligands = {}
        vina_structures = {}
        vina_ligands = {}
        placer_structures = {}
        placer_ligands = {}
        #self.AF3_structures = {}

        for chai_path, vina_path, placer_path in df[[self.chai_col, self.vina_col, self.placer_col]].values: 

            entry_name = Path(chai_path).stem
            suboutput_dir = Path(self.output_dir) / f'{entry_name}'
            suboutput_dir.mkdir(parents=True, exist_ok=True)
        
            # Chai
            if self.chai_col:
                chai_path = Path(chai_path) / "chai"
                if not chai_path.exists():
                    raise FileNotFoundError(f"Chai path not found: {chai_path}")
                
                # Read chai structure and ligand
                for cif_file in chai_path.glob("*.cif"):
                    with open(cif_file, "r") as f:
                        cif_data = pdbx.CIFFile.read(f)
                    full_structure = pdbx.get_structure(cif_data, model = 1)
                    ligand = extract_ligand(full_structure)
                    ligand.chain_id[:] = "T"
                    structure_protein_only = extract_monomer(full_structure)

                    key = cif_file.stem  # filename without extension
                    chai_structures[key] = structure_protein_only
                    chai_ligands[key] = ligand 
    
            # Vina
            if self.vina_col:
                vina_path = Path(vina_path)
                entry_name = vina_path.name
                vina_path_pdb_file = vina_path / f'{entry_name}.pdb'

                if not vina_path.exists():
                    raise FileNotFoundError(f"Vina path not found: {vina_path}")

                with open(vina_path_pdb_file, "r") as f:
                    pdb_data = pdb.PDBFile.read(f)
                full_structure = pdb.get_structure(pdb_data, model = 1)
                key = entry_name # filename without extension
                vina_structures[key] = extract_monomer(full_structure)   

                # Read vina ligand files
                vina_ligand_file_path = Path(vina_path) / f'{entry_name}-{self.ligand_name}.pdb'
                cleaned_ligand_file_path = clean_vina_ligand_file(vina_ligand_file_path)

                with open(cleaned_ligand_file_path, "r") as f:
                    vina_ligand = pdb.PDBFile.read(f)
                    vina_ligand_structure = vina_ligand.get_structure()
            
                # Iterate over the models in the vina_ligand_structure
                for model_index, model in enumerate(vina_ligand_structure, start=1):
                    # Use the model_index as the key
                    key = f"vina_ligand{model_index}"
                    vina_ligands[key] = model

            # Placer
            if self.placer_col: 
                placer_path = Path(placer_path) 
            # Recursively find all .pdb files in subdirectories
                pdb_files = list(placer_path.rglob("*.pdb"))
                for pdb_file in pdb_files: 
                    with open (pdb_file, 'r') as f: 
                        pdb_data = pdb.PDBFile.read(f)

                    # Loop through all models
                    for model in range(len(pdb_data.structure)):
                        full_structure = pdb.get_structure(pdb_data, model=model)
                        ligand = extract_ligand(full_structure)
                        ligand.chain_id[:] = "P"
                        
                        structure_protein_only = extract_monomer(full_structure)
                        structure_protein_only.chain_id[:] = "C"
            
                        key = f"{pdb_file.stem}_model{model}"
                        placer_ligands[key] = ligand
                        placer_structures[key] = structure_protein_only

                        

            
            if self.vina_col:
                depth = len(vina_ligands)
            # ADAPT!!!

            # 1. Chai vs Vina
            for chai_key, chai_structure in chai_structures.items():

                chai_structure.chain_id[:] = "A"
                chai_ligand = chai_ligands[chai_key]

                for vina_ligand_key, vina_ligand_structure in vina_ligands.items():
                
                    try:
                        # Superimpose vina onto chai
                        vina_aligned, transform, _, _ = struc.superimpose_homologs(chai_structure, list(vina_structures.values())[0])
                        vina_aligned.chain_id[:] = "B"                
                                    
                        # Align vina ligands using the same transformation
                        ligand_aligned = transform.apply(vina_ligand_structure)  # Apply the transformation
                        ligand_aligned.chain_id[:] = "V"

                    except Exception as e:
                        print(f"Failed on {chai_key} vs {vina_ligand_key}: {e}")   

                    combined = struc.concatenate([
                        chai_structure,
                        vina_aligned, 
                        chai_ligand, 
                        ligand_aligned
                        ])
                
                    # Post-processing
                    combined = truncate_residue_names(combined)
                    combined = truncate_atom_names(combined)   
                
                    # Save
                    #key_pair = f"chai_{chai_key}__{vina_ligand_key}"
                    #results[key_pair] = combined

                    # Save each combined structure as a PDB file
                    output_filenames = []

                    key = f"chai_{chai_key}__{vina_ligand_key}"
                    output_path = suboutput_dir / f"{key}.pdb"
                    pdb_file = pdb.PDBFile()
                    pdb_file.set_structure(combined)
                    pdb_file = clean_pdb_atom_names(pdb_file)
                    with open(output_path, "w") as f:
                        pdb_file.write(f)
                    
                    #output_filenames.append(output_path)


            # 2. Vina vs Vina
            vina_keys = list(vina_ligands.keys())
            protein_structure = list(vina_structures.values())[0]  # Only one shared protein
            
            for i in range(len(vina_keys)):
                key1 = vina_keys[i]
                ref_ligand = vina_ligands[key1]
                #ref_ligand = vina_ligands[key1].copy()
                ref_ligand.chain_id[:] = "T"
                ref_ligand.res_id[:] = 1
                print('***')
                print(ref_ligand)

                for j in range(i+1, len(vina_keys)):
                    key2 = vina_keys[j]
                    mov_ligand = vina_ligands[key2].copy()
                    mov_ligand = vina_ligands[key2]
                        
                    mov_ligand.chain_id[:] = "V"
                    mov_ligand.res_id[:] = 2

                    combined = struc.concatenate([protein_structure, ref_ligand, mov_ligand])
                    combined = truncate_residue_names(combined)
                    combined = truncate_atom_names(combined)

                    key = f"{key1}__{key2}"
                    output_path = suboutput_dir / f"{key}.pdb"
                    pdb_file = pdb.PDBFile()
                    pdb_file.set_structure(combined)
                    pdb_file = clean_pdb_atom_names(pdb_file)
                    with open(output_path, "w") as f:
                        pdb_file.write(f)

            # 3. Chai vs Chai
            chai_keys = list(chai_structures.keys())

            for i in range(len(chai_keys)):
                for j in range(i+1, len(chai_keys)):
                    key1, key2 = chai_keys[i], chai_keys[j]
                    try:
                        ref_structure = chai_structures[key1]
                        ref_ligand = chai_ligands[key1]
                        mov_structure = chai_structures[key2]
                        mov_ligand = chai_ligands[key2]

                        aligned, transform, _, _ = struc.superimpose_homologs(ref_structure, mov_structure)
                        aligned.chain_id[:] = "B"
                        ligand_aligned = transform.apply(mov_ligand)
                        ligand_aligned.chain_id[:] = "V"

                        ref_structure.chain_id[:] = "A"
                        ref_ligand.chain_id[:] = "T"

                        combined = struc.concatenate([ref_structure, aligned, ref_ligand, ligand_aligned])
                        combined = truncate_residue_names(combined)
                        combined = truncate_atom_names(combined)

                        key = f"chai_{key1}__chai_{key2}"
                        output_path = suboutput_dir / f"{key}.pdb"
                        pdb_file = pdb.PDBFile()
                        pdb_file.set_structure(combined)
                        pdb_file = clean_pdb_atom_names(pdb_file)
                        with open(output_path, "w") as f:
                            pdb_file.write(f)
                    except Exception as e:
                        print(f"Failed chai-chai {key1} vs {key2}: {e}")

            # 4. Chai vs Placer



            # 5. Vina vs. Placer



            # 6. Placer vs Placer


        #return output_filenames


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames += self.__execute(df_chunk, self.output_dir)
                    
                df['superimposedstructure_dir'] = output_filenames
                return df
            
            else:
                output_filenames = self.__execute(df, self.output_dir)
                df['superimposedstructure_dir'] = output_filenames
                return df
        else:
            print('No output directory provided')




                
'''
                # Concatenate
                combined = struc.concatenate([
                        chai_structure_stack,
                        vina_structure_stack,
                        chai_ligand_stack,
                        vina_ligand_stack,
                        #  ,
                        # placer_ligand_stack
                ])     
'''
