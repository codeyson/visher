"""
inspect_checkpoint.py
Run this once to print the actual config your checkpoint was trained with.
Usage: python inspect_checkpoint.py checkpoints/best_model.pth
"""
import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pth"
state = torch.load(path, map_location="cpu")

# Unwrap common wrapper keys
if isinstance(state, dict):
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            state = state[key]
            break

print("\n── Checkpoint keys & shapes ──────────────────────────────")
for k, v in state.items():
    print(f"  {k:55s} {list(v.shape)}")

print("\n── Inferred config ────────────────────────────────────────")

# nb_samp (SincConv out_channels) = first_bn.weight size
nb_samp = state["first_bn.weight"].shape[0]
print(f"  nb_samp       : {nb_samp}")

# first_conv kernel size
sinc_key = [k for k in state if "sinc_conv" in k and "low_hz_" in k]
if sinc_key:
    # low_hz_ shape = (out_channels, 1)  – kernel size inferred from n_ buffer
    n_key = [k for k in state if "n_" in k]
    if n_key:
        first_conv = state[n_key[0]].shape[1] * 2 + 1
        print(f"  first_conv    : {first_conv}")

# nb_filts – infer from res_blocks
res_keys = sorted([k for k in state if "res_blocks" in k and "conv1.weight" in k])
nb_filts = []
for rk in res_keys:
    w = state[rk]
    nb_filts.append([w.shape[1], w.shape[0]])
print(f"  nb_filts      : {nb_filts}")

# GRU hidden size
gru_key = [k for k in state if "gru.weight_ih_l0" in k]
if gru_key:
    gru_node = state[gru_key[0]].shape[0] // 3
    gru_input = state[gru_key[0]].shape[1]
    print(f"  gru_node      : {gru_node}")
    print(f"  gru_input_size: {gru_input}  (= last nb_filts channel)")

# nb_gru_layer
gru_layers = len([k for k in state if "gru.weight_ih" in k])
print(f"  nb_gru_layer  : {gru_layers}")

# fc node
fc_key = [k for k in state if k == "fc.weight"]
if fc_key:
    nb_fc = state["fc.weight"].shape[0]
    print(f"  nb_fc_node    : {nb_fc}")

# nb_classes (multi output)
multi_key = [k for k in state if "fc_multi.weight" in k]
if multi_key:
    nb_classes = state[multi_key[0]].shape[0]
    print(f"  nb_classes    : {nb_classes}")

print()