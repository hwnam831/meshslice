gcloud compute tpus tpu-vm ssh $1 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 5 5 5 5;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 8 8 8 8;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 10 10 10 10;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 20 20 20 20;"
gcloud compute tpus tpu-vm ssh $1 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 3 3 3 3;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 6 6 6 6;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 8 8 8 8;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 12 12 12 12;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 24 24 24 24;"
gcloud compute tpus tpu-vm ssh $1 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 4 --ncols 4 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 16 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 32 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 64 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 64 --ncols 16 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 96 --alg systolic --nrows 64 --ncols 32 --ksplit 16 16 16 16;"
gcloud compute tpus tpu-vm ssh $1 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 4 --ncols 4 --ksplit 4 4 4 4;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 16 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 32 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 64 --ncols 8 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 64 --ncols 16 --ksplit 16 16 16 16;\
python BenchTransformer.py --nheads 160 --alg systolic --nrows 64 --ncols 32 --ksplit 16 16 16 16;"