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


class SuperimposeStructures(Step):
    def __init__(self, structure1 = None, structure2 = None, ligand_name: str = "", output_dir: str = '', num_threads=1): 
        self.structure1 = structure1
        self.structure2 = structure2
        self.ligand_name = ligand_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_threads = num_threads or 1

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> list:

        structure1 = {}
        structure2 = {}