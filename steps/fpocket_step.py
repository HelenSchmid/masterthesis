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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Fpocket(Step):
    def __init__(self, pdb_col: str, output_dir: str, num_threads=1):
        self.pdb_col = pdb_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
        
        def run_fpocket(row):
            pdb_path = Path(row[self.pdb_col])

            if not pdb_path.exists():
                logger.warning(f"PDB file not found: {pdb_path}")
                return pd.Series({"fpocket_dir": None})

            logger.info(f"Running fpocket on {pdb_path.name}")

            # Run fpocket in the directory where the PDB file is located
            run_dir = pdb_path.parent
            print(f"run_dir: {run_dir}")

            result = subprocess.run(
                ["fpocket", "-f", str(pdb_path), "-M", '3'],
                cwd=run_dir,
                capture_output=True,
                text=True
            )
            logger.info(f"STDOUT:\n{result.stdout}")
            logger.error(f"STDERR:\n{result.stderr}")   

            if result.returncode != 0:
                logger.error(f"fpocket failed on {pdb_path.name}:\n{result.stderr}")
                return pd.Series({"fpocket_dir": None})

            # fpocket should create <name>_out folder in the same directory as the input PDB
            expected_out_dir = run_dir / f"{pdb_path.stem}_out"
            if not expected_out_dir.exists():
                logger.warning(f"fpocket output directory not found: {expected_out_dir}")
                return pd.Series({"fpocket_dir": None})

            # Move to final output location
            final_out = self.output_dir / f"{pdb_path.stem}_fpocket_out"
            if final_out.exists():
                shutil.rmtree(final_out)

            try:
                shutil.move(str(expected_out_dir), str(final_out))
            except Exception as e:
                logger.error(f"Failed to move {expected_out_dir} to {final_out}: {e}")
                return pd.Series({"fpocket_dir": None})

            return pd.Series({"fpocket_dir": str(final_out)})

        df_out = df.copy()
        result_df = df_out.apply(run_fpocket, axis=1)
        return pd.concat([df_out, result_df], axis=1)

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.num_threads > 1:
            df_chunks = np.array_split(df, self.num_threads)
            with ThreadPool(self.num_threads) as pool:
                processed = pool.map(self.__execute, df_chunks)
            return pd.concat(processed)
        else:
            return self.__execute(df)