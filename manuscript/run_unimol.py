
# Embed substrates using unimol
from enzymetk.embedchem_unimol_step import UniMol
from enzymetk.save_step import Save
import pandas as pd

# run in enzymetk environemnt
df_mmseqs2_esm2 = pd.read_pickle('/nvme2/helen/masterthesis/manuscript/esm2/promiscuous_esterases_mmseqs2_esm2.pkl')
smiles_col = 'substrates_split'
output_dir = 'unimol/'

df_mmseqs2_esm2 << (UniMol(smiles_col) >> Save(f'{output_dir}promiscuous_esterases_unimol_embedded.pkl'))
