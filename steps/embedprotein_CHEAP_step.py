from step import Step
import pandas as pd
import numpy as np
from tempfile import TemporaryDirectory
from pathlib import Path
import logging
#import h5py
from multiprocessing.dummy import Pool as ThreadPool
import os
import subprocess
import sys

sys.path.append('/home/helen/cheap_proteins/src/')

import torch
from cheap.pretrained import CHEAP_shorten_1_dim_64

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

# run in cheap environment

from step import Step
import pandas as pd
import numpy as np
import torch
from cheap.pretrained import CHEAP_shorten_1_dim_64
from tempfile import TemporaryDirectory
import logging

import seaborn as sns
import matplotlib.pyplot as plt
import einops
import tempfile


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CHEAP(Step):
    def __init__(self, id_col: str, seq_col: str, num_threads=1):
        """
        Initialize the CHEAP step.
        """
        self.id_col = id_col
        self.seq_col = seq_col
        self.num_threads = num_threads or 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True).to(self.device)

    def __execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the embedding process on the dataframe.
        """
        df[self.seq_col] = df[self.seq_col].apply(lambda x: ''.join(x) if isinstance(x, list) else str(x))

        sequences = df[self.seq_col].tolist()
        ids = df[self.id_col].tolist()

        try:
            with torch.no_grad():
                emb, mask = self.pipeline(sequences)
                emb = emb.cpu().numpy()  # Move embeddings to CPU and convert to numpy
        except Exception as e:
            logger.error(f"Error during CHEAP embedding: {e}")
            df["cheap_embedding"] = None
            return df

        # Create a dictionary of embeddings for fast lookup by ID
        embeddings = {id_: emb[i] for i, id_ in enumerate(ids)}

        # Use a vectorized approach to assign embeddings to the dataframe
        df["cheap_embedding"] = df[self.id_col].map(embeddings)
        
        print(emb.shape)
        print(mask.shape)

        return df

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the embedding process, possibly using multiple threads.
        """
        with TemporaryDirectory() as tmp_dir:
            if self.num_threads > 1:
                # Split the dataframe into chunks for parallel processing
                df_list = np.array_split(df, self.num_threads)
                
                # Process each chunk in parallel using ThreadPool
                with ThreadPool(self.num_threads) as pool:
                    output_filenames = pool.map(self.__execute, df_list)
                
                # Concatenate the results from all chunks
                df = pd.concat(output_filenames)
            else:
                # If only one thread, process the entire dataframe at once
                df = self.__execute(df)

            return df
