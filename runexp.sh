gcloud compute tpus tpu-vm ssh $1 --worker=all --command="\
python BenchTransformer.py --nrows $2 --ncols $3 --alg meshflow;\
python BenchTransformer.py --nrows $2 --ncols $3 --alg collective;\
python BenchTransformer.py --nrows $2 --ncols $3 --alg wang;\
python BenchTransformer.py --nrows $2 --ncols $3 --alg cannon;"
