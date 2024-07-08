CUDA_VISIBLE_DEVICES=0 nohup sh run.sh cola 50 snn > cola.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup sh run.sh mnli 5 snn > mnli.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh run.sh mrpc 20 snn > mrpc.out 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup sh run.sh sst2 10 snn > sst-2.out 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup sh run.sh stsb 20 snn > sts-b.out 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup sh run.sh qqp 5 snn > qqp.out 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup sh run.sh qnli 10 snn > qnli.out 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup sh run.sh rte 20 snn > rte.out 2>&1 &


