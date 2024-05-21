gcloud compute tpus tpu-vm ssh --zone "us-central2-b" $1 --project "uiuc-cs-hn5" --worker=all --command="pip install \
  --upgrade 'jax[tpu]>0.3.0' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
gcloud compute tpus tpu-vm ssh $1  --worker=all --command="pip install tensorflow tensorboard-plugin-profile"
gcloud compute tpus tpu-vm scp --worker=all *.py $1:~/