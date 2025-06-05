
import torch
import esm

# Load ESMFold model
model, alphabet = esm.pretrained.esmfold_v1()
model = model.eval()

# Prepare sequence
sequence = "MAPLRKTY"  # Replace with your sequence
batch_converter = alphabet.get_batch_converter()
data = [("protein1", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Define a hook to capture trunk embeddings
activations = {}
def get_activation(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

# Hook the last trunk layer
model.trunk.keys[-1].register_forward_hook(get_activation("trunk_last"))

# Run inference
with torch.no_grad():
    output = model.infer(batch_tokens)
    trunk_embeddings = activations["trunk_last"]  # Structural embeddings
    coords = output["positions"][-1]  # Predicted coordinates

# Print shapes
print("Trunk embeddings shape:", trunk_embeddings.shape)
print("Coordinates shape:", coords.shape)

# Optionally save to file
import numpy as np
np.save("trunk_embeddings.npy", trunk_embeddings.squeeze(0).numpy())
np.save("coords.npy", coords.squeeze(0).numpy())



'''
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")


# Load ESMFold model
model, alphabet = esm.pretrained.esmfold_v1()
model = model.eval()

# Prepare sequence
sequence = "MAPLRKTY"  # Replace with your sequence
batch_converter = alphabet.get_batch_converter()
data = [("protein1", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Define a hook to capture trunk embeddings
activations = {}
def get_activation(name):
    def hook(module, input, output):
        activations[name] = output
    return hook

# Hook the last trunk layer (adjust as needed)
model.trunk.keys[-1].register_forward_hook(get_activation("trunk_last"))

# Run inference
with torch.no_grad():
    output = model.infer(batch_tokens)
    trunk_embeddings = activations["trunk_last"]  # Structural embeddings
    coords = output["positions"][-1]  # Predicted coordinates

# Print shapes
print("Trunk embeddings shape:", trunk_embeddings.shape)
print("Coordinates shape:", coords.shape)


# Load the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
#model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", output_hidden_states=True)
#tokenizer = EsmTokenizer.from_pretrained("facebook/esmfold_v1")
#model = EsmFoldModel.from_pretrained("facebook/esmfold_v1", output_hidden_states=True)


model_name = "facebook/esmfold_v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmFoldForSequenceToStructure.from_pretrained(model_name)
model.eval()

# Optionally, uncomment to set a chunk size for axial attention. This can help reduce memory.
# Lower sizes will have lower memory requirements at the cost of increased speed.
# model.set_chunk_size(128)
# model = model.cuda() ???????????????
# model.esm = model.esm.half() # Uncomment to switch the stem to float16. For performance optimization. 

# Example protein sequence
sequence = "MAA"

# Tokenize the sequence
inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False) # convert protein sequence to tokens

# Predict the structure (outputs coordinates)
with torch.no_grad():
    outputs = model(**inputs)


hidden_states = outputs.hidden_states
# hidden_states now contains a tuple of tensors, where each tensor represents the hidden states of a layer.
last_hidden_state = hidden_states[-1]

print(last_hidden_state)'
'''