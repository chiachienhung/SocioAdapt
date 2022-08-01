gpu=$1
bert_dir=$2
output_dir=$3
train_file=$4
val_file=$5
add1=$6
add2=$7
add3=$8
add4=$9

#50K, 2K // 100K, 8K
CUDA_VISIBLE_DEVICES=$gpu python run_mtl_cls.py \
    --model_name_or_path=${bert_dir} \
    --output_dir=${output_dir} \
    --train_file=${train_file} \
    --validation_file=${val_file} \
    --cache_dir="" \
    --line_by_line \
    --do_train \
    --do_eval \
    --load_best_model_at_end \
    --max_train_samples=100000 \
    --max_eval_samples=8000\
    --learning_rate=1e-5 --evaluation_strategy="epoch"\
    --save_strategy="epoch"\
    --save_steps=2000 --logging_steps=100 --gradient_accumulation_steps=2\
    --num_train_epochs=30 --save_total_limit 2\
    --per_device_train_batch_size=16 --per_device_eval_batch_size=8 --max_seq_length=128 --patience=3\
    ${add1} ${add2} ${add3} ${add4}