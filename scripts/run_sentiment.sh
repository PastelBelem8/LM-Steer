#!/bin/bash
TRIAL=sentiment-$3
mkdir -p logs/$TRIAL

## #################### ################### #########################
## Training:
## bash scripts/run_sentiment.sh train "0" "gpt2-large"
## bash scripts/run_sentiment.sh train "0" "meta-llama/Llama-3.2-3B"
##
## Generate:
## bash scripts/run_sentiment.sh generate "0" "gpt2-large" neutral 5
## ^Note: original code does not work with low_resource_mode
## bash scripts/run_sentiment.sh generate "0" "meta-llama/Llama-3.2-3B" neutral 5
##
## Evaluate:
## bash scripts/run_sentiment.sh evaluate "0" "gpt2-large" neutral 5
##
## 
## #################### ################### #########################
if [ "$1" = "train" ]; then
    echo "TRAINING"
    CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=. python experiments/training/train.py \
        --dataset_name sentiment-sst5 \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $3 --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 16 --max_length 256 \
        --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3
    echo "Finished training"
elif [ "$1" = "generate" ]; then
    SOURCE=$4
    CONTROL=$5
    echo "model_name=$3; source=$4; control=$5; logs_dir=$TRIAL"
    { time ( \
        CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=. python experiments/training/generate.py \
        --eval_file data/prompts/sentiment_prompts-10k/${SOURCE}_prompts.jsonl \
        --output_file logs/$TRIAL/preds__${SOURCE}__control_${CONTROL}.jsonl \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $3 --cuda  \
        --adaptor_class multiply --num_steers 2 --rank 1000 \
        --max_length 256 --verbose --steer_values ${CONTROL} 1 --top_p 0.9
    ); } 2>&1 | tee logs/$TRIAL/runtime__${1}__preds__${SOURCE}__control_${CONTROL}.txt

elif [ "$1" = "evaluate" ]; then
    SOURCE=$4
    CONTROL=$5
    echo "model_name=$3; source=$4; control=$5; logs_dir=logs/$TRIAL"
    { time ( \
        CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=. python experiments/evaluation/evaluate.py \
        --generations_file logs/$TRIAL/preds__${SOURCE}__control_${CONTROL}.jsonl \
        --metrics sentiment,ppl-big,dist-n \
        --output_file evals__${SOURCE}__control_${CONTROL}.txt
    ); } 2>&1 | tee logs/$TRIAL/runtime__${1}__evals__${SOURCE}__control_${CONTROL}.txt
fi
