from PIL import Image
import numpy as np
from time import time
import pickle
from tqdm import tqdm
import glob
import string
import os

import tensorflow.keras.preprocessing.image
import tensorflow.keras.applications.mobilenet
import tensorflow.keras.applications.inception_v3
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import (LSTM, Embedding, Dense, Dropout)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model
from tensorflow.keras import Input

START = "startseq"
STOP = "endseq"
EPOCHS = 1

# Initialize
root_captioning = "./data/captions"
null_punct = str.maketrans('', '', string.punctuation)
lookup = dict()

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"

with open(os.path.join(root_captioning, 'Flickr8k_text',
                       'Flickr8k.token.txt'), 'r') as fp:
    max_length = 0
    for line in fp.read().split('\n'):
        tok = line.split()
        if len(line) >= 2:
            id = tok[0].split('.')[0]
            desc = tok[1:]

            # Cleanup description
            desc = [word.lower() for word in desc]
            desc = [w.translate(null_punct) for w in desc]
            desc = [word for word in desc if len(word) > 1]
            desc = [word for word in desc if word.isalpha()]
            max_length = max(max_length, len(desc))

            if id not in lookup:
                lookup[id] = list()
            lookup[id].append(' '.join(desc))

lex = set()
for key in lookup:
    [lex.update(d.split()) for d in lookup[key]]

img = glob.glob(os.path.join(root_captioning, 'Flicker8k_Dataset', '*.jpg'))

train_images_path = os.path.join(root_captioning,
                                 'Flickr8k_text', 'Flickr_8k.trainImages.txt')
train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
test_images_path = os.path.join(root_captioning,
                                'Flickr8k_text', 'Flickr_8k.testImages.txt')
test_images = set(open(test_images_path, 'r').read().strip().split('\n'))

train_img = []
test_img = []

for i in img:
    f = os.path.split(i)[-1]
    if f in train_images:
        train_img.append(f)
    elif f in test_images:
        test_img.append(f)

train_descriptions = {k: v for k, v in lookup.items() if f'{k}.jpg'
                      in train_images}
for n, v in train_descriptions.items():
    for d in range(len(v)):
        v[d] = f'{START} {v[d]} {STOP}'

encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input


def encodeImage(img):
    img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    x = tensorflow.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    x = encode_model.predict(x) 
    x = np.reshape(x, OUTPUT_DIM)
    return x

train_path = os.path.join(root_captioning, "data", f'train{OUTPUT_DIM}.pkl')
if not os.path.exists(train_path):
    start = time()
    encoding_train = {}
    for id in tqdm(train_img):
        image_path = os.path.join(root_captioning, 'Flicker8k_Dataset', id)
        img = tensorflow.keras.preprocessing.image.load_img(image_path,
                                                            target_size=(HEIGHT, WIDTH))
        encoding_train[id] = encodeImage(img)
    with open(train_path, "wb") as fp:
        pickle.dump(encoding_train, fp)
    print(f"\nGenerating training set took: {hms_string(time()-start)}")
else:
    with open(train_path, "rb") as fp:
        encoding_train = pickle.load(fp)

test_path = os.path.join(root_captioning, "data", f'test{OUTPUT_DIM}.pkl')
if not os.path.exists(test_path):
    start = time()
    encoding_test = {}
    for id in tqdm(test_img):
        image_path = os.path.join(root_captioning, 'Flicker8k_Dataset', id)
        img = tensorflow.keras.preprocessing.image.load_img(image_path,
                                                            target_size=(HEIGHT, WIDTH))
        encoding_test[id] = encodeImage(img)
    with open(test_path, "wb") as fp:
        pickle.dump(encoding_test, fp)
    print(f"\nGenerating testing set took: {hms_string(time()-start)}")
else:
    with open(test_path, "rb") as fp:
        encoding_test = pickle.load(fp)

all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1

vocab_size = len(idxtoword) + 1
vocab_size
max_length += 2

def data_generator(descriptions, photos, wordtoidx,
                   max_length, num_photos_per_batch):
    x1, x2, y = [], [], []
    n = 0
    while True:
        for key, desc_list in descriptions.items():
            n += 1
            photo = photos[key+'.jpg']
            for desc in desc_list:
                seq = [wordtoidx[word] for word in desc.split(' ')
                       if word in wordtoidx]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical(
                        [out_seq], num_classes=vocab_size)[0]
                    x1.append(photo)
                    x2.append(in_seq)
                    y.append(out_seq)
            if n == num_photos_per_batch:
                yield ([np.array(x1), np.array(x2)], np.array(y))
                x1, x2, y = [], [], []
                n = 0

glove_dir = os.path.join(root_captioning, 'glove.6B')
embeddings_index = {}

f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in wordtoidx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)
caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

number_pics_per_bath = 3
steps = len(train_descriptions)//number_pics_per_bath

model_path = os.path.join(root_captioning, "data", f'caption-model.hdf5')
if not os.path.exists(model_path):
    for i in tqdm(range(EPOCHS*2)):
        generator = data_generator(train_descriptions, encoding_train,
                                   wordtoidx, max_length, number_pics_per_bath)
        caption_model.fit_generator(generator, epochs=1,
                                    steps_per_epoch=steps, verbose=1)

    caption_model.optimizer.lr = 1e-4
    number_pics_per_bath = 6
    steps = len(train_descriptions)//number_pics_per_bath

    for i in range(EPOCHS):
        generator = data_generator(train_descriptions, encoding_train,
                                   wordtoidx, max_length, number_pics_per_bath)
        caption_model.fit_generator(generator, epochs=1,
                                    steps_per_epoch=steps, verbose=1)
    caption_model.save_weights(model_path)
else:
    caption_model.load_weights(model_path)

def generateCaption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def make_prediction(image):
    image = encodeImage(image).reshape((1, OUTPUT_DIM))
    return generateCaption(image)
