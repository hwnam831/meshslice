let "BSIZE=$1 * $2"
#echo $BSIZE
gcloud compute tpus tpu-vm ssh tpu4x4 --worker=all --command="\
python BenchTransformer.py --nrows $1 --ncols $2 --alg noff;\
python BenchTransformer.py --nrows $1 --ncols $2 --alg collective;\
python BenchTransformer.py --nrows $1 --ncols $2 --alg wang;"
