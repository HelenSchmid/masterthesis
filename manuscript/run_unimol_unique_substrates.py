
# Embed substrates using unimol
from enzymetk.embedchem_unimol_step import UniMol
from enzymetk.save_step import Save
import pandas as pd

# run in enzymetk environemnt
df_mmseqs2_esm2 = pd.read_pickle('/nvme2/helen/masterthesis/manuscript/promiscuous_esterases_mmseqs2_esm2.pkl')
smiles_col = 'substrates_split'
output_dir = 'unimol/'

# Remove rows where any SMILES string in the list contains '*' or is '[H+]'. These cannot be embedded using unimol. 
def is_valid_smiles(smi):
    if not isinstance(smi, str):
        return False
    return '*' not in smi and smi.strip() != '[H+]'

df_mmseqs2_esm2_unimol_reduced = df_mmseqs2_esm2[df_mmseqs2_esm2['substrates_split'].apply(is_valid_smiles)].reset_index(drop=True)
unique_substrates_df = pd.DataFrame(df_mmseqs2_esm2_unimol_reduced['substrates_split'].unique(), columns=['substrates_split'])

unique_substrates_df << (UniMol(smiles_col) >> Save(f'{output_dir}promiscuous_esterases_unique_substrates_unimol_embedded.pkl'))
