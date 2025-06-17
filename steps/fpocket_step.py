from steps.step import Step
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess

import shutil
import uuid

# How to run fpocket in terminal: fpocket -f /home/helen/cec_degrader/generalize/alphafold_structures/A1RRK1_structure.pdb
# -r string: (None) This parameter allows you to run fpocket in a restricted mode. Let's suppose you have a very shallow or large pocket with a ligand inside and the automatic pocket prediction always splits up you pocket or you have only a part of the pocket found. Specifying your ligand residue with -r allows you to detect and characterize you ligand binding site explicitly. 
# For instance for `1UYD.pdb` you can specify `-r 1224:PU8:A` (residue number of the ligand: residue name of the ligand: chain of the ligand)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def extract_fpocket_features(): 
    




class Fpocket(Step):
    def __init__(self, preparedfiles_dir: str = 'filteringpipeline/preparedfiles', output_dir: str = '', num_threads: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preparedfiles_dir = Path(preparedfiles_dir)
        self.num_threads = num_threads if num_threads > 0 else 1 # Ensure at least 1 thread

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:

        results_list = []

        if not self.preparedfiles_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.preparedfiles_dir}")

        for _, row in df.iterrows():
            best_structure_name = row['best_structure']
            pdb_file_path = self.preparedfiles_dir / f"{best_structure_name}.pdb"
        
            # Run fpocket execution in temporary directory
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Copy the PDB file to the temporary directory for fpocket to process it there
                temp_pdb_path = temp_path / pdb_file_path.name
                try:
                    shutil.copy(str(pdb_file_path), str(temp_pdb_path))
                except FileNotFoundError:
                    logger.error(f"PDB file not found for copying: {pdb_file_path}")
                    return None
                except Exception as e:
                    logger.error(f"Error copying PDB file {pdb_file_path} to {temp_pdb_path}: {e}")
                    return None

                logger.info(f"Running fpocket on {temp_pdb_path} in {temp_dir}")
                result = subprocess.run(
                    ["fpocket", "-f", str(temp_pdb_path), "-r", "1:LIG:B"],
                    cwd=temp_path, # Run fpocket within the temporary directory
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    logger.error(f"fpocket failed on {pdb_file_path.name} in {temp_path}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
                    return None

                # fpocket creates <name>_out folder and <name>.log in the directory it was run.
                expected_out_dir_in_temp = temp_path / f"{temp_pdb_path.stem}_out"
                if not expected_out_dir_in_temp.exists():
                    logger.warning(f"fpocket output directory not found in temp dir: {expected_out_dir_in_temp}")
                    return None
                
                # Define final output path for this specific run
                final_out_dir = self.output_dir / f"{pdb_file_path.stem}_fpocket_output"
                
                # Clean up any existing previous output for this structure
                if final_out_dir.exists():
                    logger.warning(f"Existing fpocket output for {pdb_file_path.stem} found at {final_out_dir}. Removing old fpocket output.")
                    shutil.rmtree(final_out_dir)

                try:
                    # Move the entire output folder from temp_dir to the final self.output_dir
                    shutil.move(str(expected_out_dir_in_temp), str(final_out_dir))
                    # Also move the log file if it exists
                    expected_log_file_in_temp = temp_path / f"{temp_pdb_path.stem}.log"
                    if expected_log_file_in_temp.exists():
                        shutil.move(str(expected_log_file_in_temp), str(final_out_dir / expected_log_file_in_temp.name))
                    results_list.append(final_out_dir)

                except Exception as e:
                    logger.error(f"Failed to move {expected_out_dir_in_temp} to {final_out_dir}: {e}")
                    return None

        logger.info(f"Successfully processed {pdb_file_path.name}. Output in: {final_out_dir}")
        return results_list


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.output_dir:
            fpocket_dir = self.__execute(df)
            df['fpocket_dir'] = fpocket_dir

            return df
        else:
            print('No output directory provided')

