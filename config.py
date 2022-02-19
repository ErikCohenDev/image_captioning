import tensorflow as tf

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False

ROOT_CAPTIONING = None
START_TOKEN = "startseq"
STOP_TOKEN = "endseq"
EPOCHS = 1
USE_INCEPTION = True
# Path to the images
IMAGES_PATH = "Flicker8k_Dataset"

# Desired image dimensions
IMAGE_SIZE = (299, 299)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 25

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 1
AUTOTUNE = tf.data.AUTOTUNE

# Hyperparameters
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 1000

if COLAB:
    ROOT_CAPTIONING = "/content/drive/My Drive/projects/captions"
else:
    ROOT_CAPTIONING = "./data/captions"
