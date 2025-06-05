from generalize.step import Step
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

# WRONG!!! WE NEED PROSTT5 AND NOT PROTT5!!!! DON'T CONTINUE WORKING WITH THIS RIGHT NOW. 

## python prott5-extract.py  --input sequences/some.fasta --output embeddings/protein_embeddings.h5 --per_protein 1


class ProtT5(Step):
    
    def __init__(self, id_col: str, seq_col: str, num_threads=1, tmp_dir: str = None):
        self.seq_col = seq_col
        self.id_col = id_col
        self.num_threads = num_threads or 1
        self.tmp_dir = tmp_dir

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> pd.DataFrame:
    
        input_filename = f'{tmp_dir}input.fasta'
    
        result = subprocess.run(
            ['python', Path(__file__).parent/'prott5-extract.py', '--input', 
             input_filename, '--output', tmp_dir, 'per_protein 1'], 
             capture_output=True, text=True)
        
        print(result)

        df['prott5_embedding'] = result
        return df


    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        with TemporaryDirectory() as tmp_dir:
            if self.num_threads > 1:
                output_filenames = []
                df_list = np.array_split(df, self.num_threads)
                for df_chunk in df_list:
                    output_filenames.append(self.__execute(df_chunk, tmp_dir))
                    
                df = pd.DataFrame()
                for tmp_df in output_filenames:
                    df = pd.concat([df, tmp_df])
                return df
            
            else:
                return self.__execute(df, tmp_dir)