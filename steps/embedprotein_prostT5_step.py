from step import Step
import pandas as pd
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
import numpy as np
import h5py
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

# run in prostT5 environment
# python embed.py --input /path/to/your/input_sequences.fasta --output /path/to/your/output_embeddings.h5 --model "Rostlab/ProstT5" --half 1 --is_3Di 0 --per_protein 1

class ProstT5(Step): 
    
    def __init__(self, id_col: str, seq_col: str, num_threads=1, tmp_dir: str = None):
        self.seq_col = seq_col
        self.id_col = id_col
        self.num_threads = num_threads or 1
        self.tmp_dir = tmp_dir

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> pd.DataFrame:
    
        output_filename = os.path.join(tmp_dir, "embeddings.h5")
        input_filename = os.path.join(tmp_dir, "input.fasta")

        with open(input_filename, "w") as fasta_file:
            for index, row in df.iterrows():
                sequence = f">{row[self.id_col]}\n{row[self.seq_col]}\n"
                fasta_file.write(sequence)

        result = subprocess.run(
        ['python', Path(__file__).parent/'prostT5_extract.py', '--input', 
        input_filename, '--output', output_filename, '--model', 'Rostlab/ProstT5', '--per_protein', '1'], 
        capture_output=True, text=True)

        print("Subprocess stdout:", result.stdout)
        print("Subprocess stderr:", result.stderr)
        print(result)

        if result.returncode != 0:
            logger.error(f"Subprocess failed: {result.stderr}")
            raise RuntimeError(f"Embedding extraction failed:\n{result.stderr}")

        embeddings = {}
        try:
            with h5py.File(output_filename, "r") as hf:
                for sequence_id in hf:
                    embeddings[sequence_id] = np.array(hf[sequence_id])
                    print(sequence_id)

            df["prostT5_embedding"] = df[self.id_col].apply(lambda x: embeddings.get(x, None))

        except FileNotFoundError:
            print(f"Error: HDF5 file '{output_filename}' not found.")
            df["prostT5_embedding"] = None

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