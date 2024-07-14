gcloud compute tpus tpu-vm ssh $1 --command="\
python Autotuner.py --nrows 8 --ncols 4 --batchsize 32 > gpt3-32.log;\
python Autotuner.py --nrows 8 --ncols 8 --batchsize 64 > gpt3-64.log;\
python Autotuner.py --nrows 16 --ncols 8 --batchsize 128 > gpt3-128.log;\
python Autotuner.py --nrows 16 --ncols 16 --batchsize 256 > gpt3-256.log;\
python Autotuner.py --nrows 32 --ncols 16 --batchsize 512 > gpt3-512.log;\
python Autotuner.py --nrows 32 --ncols 32 --batchsize 1024 > gpt3-1024.log;\
python Autotuner.py --nrows 32 --ncols 64 --batchsize 2048 > gpt3-2048.log;\
python Autotuner.py --nrows 8 --ncols 4 --batchsize 32 --nheads 160 > megatron-32.log;\
python Autotuner.py --nrows 8 --ncols 8 --batchsize 64 --nheads 160 > megatron-64.log;\
python Autotuner.py --nrows 16 --ncols 8 --batchsize 128 --nheads 160 > megatron-128.log;\
python Autotuner.py --nrows 16 --ncols 16 --batchsize 256 --nheads 160 > megatron-256.log;\
python Autotuner.py --nrows 32 --ncols 16 --batchsize 512 --nheads 160 > megatron-512.log;\
python Autotuner.py --nrows 32 --ncols 32 --batchsize 1024 --nheads 160 > megatron-1024.log;\
python Autotuner.py --nrows 32 --ncols 64 --batchsize 2048 --nheads 160 > megatron-2048.log;"