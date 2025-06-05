import os
import pandas as pd
from pathlib import Path
import logging
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from steps.step import Step
import re
from tempfile import TemporaryDirectory

from Bio.PDB import MMCIFParser
from Bio.PDB import PDBIO
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_cif_to_pdb(cif_filepath, pdb_filepath=None):
    """
    Converts a mmCIF file to a PDB file using Biopython.
    """
    cif_filepath = Path(cif_filepath)

    if pdb_filepath is None:
        pdb_filepath = cif_filepath.with_suffix('.pdb')
    else:
        pdb_filepath = Path(pdb_filepath)

    parser = MMCIFParser()
    try:
        structure_id = cif_filepath.stem
        structure = parser.get_structure(structure_id, str(cif_filepath))

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_filepath))
        print(f"Successfully converted '{cif_filepath}' to '{pdb_filepath}'")
        return pdb_filepath
    except Exception as e:
        print(f"Error converting {cif_filepath}: {e}")
        return None

class PrepareVina(Step):
    def __init__(self, chai_dir = None,  output_dir: str = '' , num_threads=1):
        self.chai_dir = chai_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1
        

    def __execute(self, df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
        results = []

        for vina_path in df[self.vina_dir].apply(str):

            vina_path = Path(vina_path)
            if not vina_path.exists():
                logger.warning(f"Vina path not found: {vina_path}")
                results.append(None)
                continue

            entry_name = vina_path.stem
            ligand_file = vina_path.parent / f'{entry_name}-{self.ligand_name}.pdb'

            try:
                # Read and clean ligand
                cleaned_ligand_file = clean_vina_ligand_file(ligand_file)

                # Combine protein and ligands in same file
                output_files = split_ligands_and_combine(
                    protein_path=vina_path,
                    ligands_path=cleaned_ligand_file,
                    entry_name=entry_name,
                    output_dir=self.output_dir, 
                    renumber_atoms=True
                )
                results.append(output_files)

            except Exception as e:
                logger.error(f"Error processing {vina_path}: {e}")
                results.append(None)

        return results


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames += self.__execute(df_chunk, self.output_dir)
                    
                df['chai_files_for_superimposition'] = output_filenames
                return df
            
            else:
                output_filenames = self.__execute(df, self.output_dir)
                df['chai_files_for_superimposition'] = output_filenames
                return df
        else:
            print('No output directory provided')