gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="python BenchTransformer.py --nheads 96 --alg noff --batchsize 64; python BenchTransformer.py --nheads 96 --alg collective --batchsize 64; python BenchTransformer.py --nheads 96 --alg cannon --batchsize 64;python BenchTransformer.py --nheads 96 --alg wang --batchsize 64;python BenchTransformer.py --nheads 96 --alg systolic --batchsize 64;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg noff --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 160 --alg collective --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 160 --alg wang --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 64 --nrows 4 --ncols 8;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg noff --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 96 --alg collective --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 96 --alg wang --batchsize 64 --nrows 4 --ncols 8;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 64 --nrows 4 --ncols 8;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg noff --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg collective --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg wang --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 64 --nrows 8 --ncols 16;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg noff --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg collective --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg wang --batchsize 64 --nrows 8 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 64 --nrows 8 --ncols 16;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 96 --alg noff --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg collective --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg cannon --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg wang --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 96 --alg systolic --batchsize 64 --nrows 16 --ncols 16;"
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nheads 160 --alg noff --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg collective --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg cannon --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg wang --batchsize 64 --nrows 16 --ncols 16;\
python BenchTransformer.py --nheads 160 --alg systolic --batchsize 64 --nrows 16 --ncols 16;"