#!/bin/bash

# default values
root_path=./datasets/
gpu=0



######################## Table 2 ###########################
nohup python -u run_periref.py \
    --root_path $root_path \
    --gpu $gpu > "logs/periref.log" &




######################## Table 3 ###########################
# nohup python -u run_periref_generic_random.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/generic_random.log" &


# nohup python -u run_periref_generic_learnable.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/generic_learnable.log" &


# nohup python -u run_periref_scale.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/scale.log" &


# nohup python -u run_periref_embedding.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/embedding.log" &





######################## Table 4 ###########################
# nohup python -u run_harmonic_regression.py --root_path $root_path > "logs/regression.log" &

# nohup python -u run_seasonal_climatology.py --root_path $root_path > "logs/climatology.log" &


# nohup python -u run_periref_mlp_x.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/xmlp.log" &


# nohup python -u run_periref_fourier_phaseonly.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/fourier_phaseonly.log" &


# nohup python -u run_periref_fourier_pointwise.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/fourier_pointwise.log" &






######################## Table 5 ###########################
# nohup python -u run_periref_perturb.py \
#     --root_path $root_path \
#     --gpu $gpu > "logs/perturb.log" &






######################## ACF Diagnostics ###########################
# nohup python -u run_acf_diagnostics.py --root_path $root_path > "logs/acf.log" &