gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 2048 --nrows 256 --ncols 8;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 2048 --nrows 128 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 2048 --nrows 64 --ncols 32;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 2048 --nrows 32 --ncols 64;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 2048 --nrows 16 --ncols 128;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 2048 --nrows 256 --ncols 8;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 2048 --nrows 128 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 2048 --nrows 64 --ncols 32;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 2048 --nrows 32 --ncols 64;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 2048 --nrows 16 --ncols 128;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 6 6 6 6;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 12 12 12 12;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 16 16 16 16;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 6 6 6 6;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 12 12 12 12;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 256 --nrows 32 --ncols 8 --ksplit 16 16 16 16;"