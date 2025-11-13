#!/bin/bash
MODE=$1
CUDA_DEVICES=$2
MODEL_NAME=$3
DATASET_NAME=$4
DATASET_GRANULARITY=$5

TRIAL=$MODEL_NAME--$DATASET_GRANULARITY--$DATASET_NAME
mkdir -p logs/$TRIAL

## #################### ################### #########################
## Training Commands
## #################### ################### #########################
## Syntax:
##    bash scripts/run_readability.sh <MODE> <CUDA_DEVICE> <MODEL_NAME> <DATASET> <DATASET_GRANULARITY> <PROMPT_COL>
## 
## Experiments Command log:
## - bash scripts/run_readability.sh train "0" "meta-llama/Llama-3.2-3B-Instruct" "ova__elem_vs_no_elem" "doc-level" "prompt"
## - bash scripts/run_readability.sh train "0" "meta-llama/Llama-3.2-3B-Instruct" "ovo__elem_vs_no_elem" "doc-level" "prompt"
##
## Debugging:
## - bash scripts/run_readability.sh train "0" "meta-llama/Llama-3.2-3B" "elem_vs_no_elem" "sent-level"
## - bash scripts/run_readability.sh generate "0" "meta-llama/Llama-3.2-3B" "elem_vs_no_elem" "sent-level" -2 "scienceqa-validation" "/home/cbelem/projects/LM-Steer/data/prompts/readability/default_scienceqa_val__prompts.jsonl"
## #################### ################### #########################
## Generate:
## #################### ################### #########################
## Syntax:
##    bash scripts/run_readability.sh <MODE> <CUDA_DEVICE> <MODEL_NAME> <DATASET> <DATASET_GRANULARITY> <CONTROL> <EVAL_NAME> <EVAL_FILEPATH>
## Command log:
## - bash scripts/run_readability.sh generate "0" "meta-llama/Llama-3.2-3B-Instruct" "elem_vs_no_elem" "doc-level" 0 "scienceqa-validation" "/home/cbelem/projects/LM-Steer/data/prompts/readability/default_scienceqa_val__prompts.jsonl"
## - bash scripts/run_readability.sh generate "0" "meta-llama/Llama-3.2-3B-Instruct" "elem_vs_no_elem" "sent-level" 5 "scienceqa-validation" "/home/cbelem/projects/LM-Steer/data/prompts/readability/default_scienceqa_val__prompts.jsonl"
## #################### ################### #########################
if [ "$MODE" = "train" ]; then
    export PROMPT_COL=$6
    FILEPATH="readability-/home/cbelem/projects/llm-readability-customization/data/eli-why/steering/$DATASET_GRANULARITY/human_annotated_train__$DATASET_NAME.csv"
    echo "TRAINING"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES PYTHONPATH=. python experiments/training/train.py \
        --dataset_name $FILEPATH \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $MODEL_NAME --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 1024 \
        --n_steps 300 --lr 1e-2 --regularization 1e-4 --epsilon 1e-3
    echo "Finished training"
elif [ "$MODE" = "generate" ]; then
    CONTROL=$6
    EVAL_NAME=$7
    EVAL_FILEPATH=$8
    # export PROMPT_COL="messages_str"
    export PROMPT_COL="formatted_example"
    export PROMPT_NUM=5
    export PROMPT_NUM_TOKENS=20
    # --temperature 0 --top_p 0
    # TODO: Set environment variables to modify the number of tokens to generate and 
    # generation configs (e.g., 1 single greedy generation)
    { time ( \
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICES PYTHONPATH=. python experiments/training/generate.py \
        --eval_file $EVAL_FILEPATH \
        --output_file logs/$TRIAL/preds__${EVAL_NAME}__control_${CONTROL}.jsonl \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $MODEL_NAME --cuda  \
        --adaptor_class multiply --num_steers 2 --rank 1000 \
        --max_length 256 --verbose --steer_values ${CONTROL} 1 --top_p 0.9
    ); } 2>&1 | tee logs/$TRIAL/runtime__${MODE}__preds__${EVAL_NAME}__control_${CONTROL}.txt

elif [ "$MODE" = "evaluate" ]; then
    echo "'EVALUATE' NOT IMPLEMENTED"
    exit -1
fi
