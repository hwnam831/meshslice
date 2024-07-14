gcloud compute tpus tpu-vm ssh $1 --worker=all --command="pip install \
  --upgrade 'jax[tpu]' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud compute tpus tpu-vm ssh $1  --worker=all --command="pip install tensorflow tensorboard-plugin-profile"
gcloud compute tpus tpu-vm scp --worker=all *.py $1:~/