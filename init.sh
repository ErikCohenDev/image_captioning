apt-get update
apt-get install -y python3-pip git unzip
echo "cloning repo"
git clone https://github.com/ErikCohenDev/neural_nets.git app
cd app
echo "installing requirements"
pip3 install --upgrade pip

# Install Tensorflow and verify Install
pip3 install --user --upgrade tensorflow 
python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

pip3 install -r requirements.txt
export PATH="$HOME/.local/bin:$PATH"

gsutil cp gs://image_caption_bucket/V04/data.zip .
unzip data.zip

echo "starting app"

nohup streamlit run app.py --server.port=8080 --server.address=0.0.0.0 &

curl localhost:8080