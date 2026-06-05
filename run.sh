#!/bin/bash

# default values
mode=boost                # options:[boost, lookback, ablation]
model=PeriRef        # options:[PeriRef, iTransformer, CycleNet]
dataset=weather2k         # options:[traffic, ecl, weather2k, soil]
gpu=0
root_path=/home/yl/datasets/
# only for mode=ablation
ablation_type=2


extra_args=()


# parse arguments like mode=boost model=iTransformer dataset=weather2k gpu=0 short_periods=3,6,12 d_ff=512
for arg in "$@"; do
    case $arg in
        mode=*) mode="${arg#*=}" ;;
        model=*) model="${arg#*=}" ;;
        dataset=*) dataset="${arg#*=}" ;;
        gpu=*) gpu="${arg#*=}" ;;
        root_path=*) root_path="${arg#*=}" ;;
        short_periods=*) short_periods="${arg#*=}" ;;
        long_period=*) long_period="${arg#*=}" ;;
        d_ff=*) d_ff="${arg#*=}" ;;
        *) extra_args+=("$arg") ;;
    esac
done

case "$mode" in
    boost)
        entry=run_boost.py
        ;;
    ablation)
        entry=run_ablation.py
        ;;
    lookback)
        entry=run_lookback.py
        ;;
    *)
        echo "Unknown mode: $mode"
        echo "Supported modes: boost, ablation, lookback"
        exit 1
        ;;
esac

cmd=(python "$entry" \
    --root_path "$root_path" \
    --model "$model" \
    --dataset "$dataset" \
    --gpu "$gpu")

if [ "$mode" = "ablation" ]; then
    [ -n "$ablation_type" ] && cmd+=(--ablation_type "$ablation_type")
fi

# append any extra unknown arguments to the command
for a in "${extra_args[@]}"; do
    cmd+=("$a")
done

nohup "${cmd[@]}" > "logs/${dataset}_${gpu}.log" 2>&1 &

echo "Started $entry with mode=$mode model=$model dataset=$dataset gpu=$gpu"
