export TASK_NAME=$1

python run_glue_no_trainer.py \
  --model_name_or_path bert-base-uncased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs $2 \
  --output_dir ./res/$TASK_NAME/$3/ \
  --seed 42 \
  --lr_scheduler_type constant \

