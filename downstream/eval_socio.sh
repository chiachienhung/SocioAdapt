gpu=$1
model_dir=$2
output_dir=$3
dataset_name=$4
add1=$5
add2=$6
add3=$7
add4=$8

CUDA_VISIBLE_DEVICES=$gpu python run-socio.py \
    --model_name_or_path=${model_dir} \
    --output_dir=${output_dir} \
    --dataset_name=${dataset_name} \
    --cache_dir="" \
    --load_best_model_at_end\
    --do_train \
    --do_eval \
    --do_predict \
    --learning_rate=5e-5 --evaluation_strategy="epoch"\
    --save_strategy="epoch"\
    --save_steps=2500 --logging_steps=100 --gradient_accumulation_steps=1\
    --num_train_epochs=20 --save_total_limit=2 --patience=5\
    --per_device_train_batch_size=32 --per_device_eval_batch_size=32\
    --metric_for_best_model="f1_macro" --remove_unused_columns=False\
    --max_seq_length=128 --overwrite_output_dir\
    ${add1} ${add2} ${add3} ${add4}