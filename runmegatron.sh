let "BSIZE=$1 * $2"
#echo $BSIZE
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nrows $1 --ncols $2 --alg noff --nheads 160;\
python BenchTransformer.py --nrows $1 --ncols $2 --alg collective --nheads 160;\
python BenchTransformer.py --nrows $1 --ncols $2 --alg wang --nheads 160;"
