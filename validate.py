import os
import pickle
import nltk
from main import generateCaption, lookup

root_captioning = "./data/captions"
OUTPUT_DIM = 2048

test_images_path = os.path.join(root_captioning,'Flickr8k_text', 'Flickr_8k.testImages.txt')
test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
test_path = os.path.join(root_captioning, "data", f'test{OUTPUT_DIM}.pkl')


with open(test_path, "rb") as fp:
    encoding_test = pickle.load(fp)

cumulative_bleu_score = 0
for image_id in encoding_test:
    image_binary = encoding_test.get(image_id).reshape((1, OUTPUT_DIM))
    predicted_caption = generateCaption(image_binary)
    image_id_getter = image_id.split('.')[0]
    expected_caption = lookup.get(image_id_getter)
    bleu_score = nltk.translate.bleu_score.sentence_bleu(
        [e.split() for e in expected_caption],
        predicted_caption.split()
    )
    cumulative_bleu_score += bleu_score

print('cumulative_bleu_score: ', cumulative_bleu_score / len(encoding_test.keys()))
