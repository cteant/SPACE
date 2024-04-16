EXPERIMENT_NAME="llama2-7b"
DATESTR=$(date +"%m-%d-%H-%M")
EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
SAVE_PATH=output_dir/${EXPERIMENT_NAME}
# alpaca_gpt4_en,lima
# alpaca_gpt4_en,alpaca_gpt4_zh,lima,oaast_sft,code_alpaca_20k,open_platypus
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_config_8_nodes.yaml src/train_bash.py \

accelerate launch --config_file accelerate_config_8_nodes.yaml src/train_bash.py \
    --stage sft \
    --model_name_or_path /huawei-data/FM/yihanling/checkpoints/Llama-2-7b-hf \
    --do_train \
    --dataset alpaca_gpt4_en,lima \
    --template default \
    --finetuning_type full \
    --output_dir $SAVE_PATH \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_strategy epoch \
    --learning_rate 5e-5 \
    --num_train_epochs 2.0 \
    --plot_loss \
    --bf16 true \
    --cutoff_len 2048 \
    --overwrite_output_dir \
    --mask_num 5 \
    --remove_unused_columns False \
    --mask_random False \
    --mask_prob 0.5 \
    --mask_efficient_train False \
    --mask_id 32002 \
    --mask_loss_weight 1
