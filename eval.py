import numpy as np
import argparse
import os
import keras
from keras.preprocessing import sequence
from preprocessing import preprocess_image, imgage_ft, parse_data
from keras.models import load_model
from keras.models import Model
import pandas as pd
import pickle
import pdb

from keras.applications.inception_v3 import InceptionV3
'''
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size = (299,299))
    array_image = image.img_to_array(img)
    array_image = np.expand_dims(array_image,axis = 0)
    array_image /=255.
    array_image -=0.5
    array_image *=2.
    
    return array_image
    

def imgage_ft(image,model):
    image = preprocess_image(image)
    encode = model.predict(image)
    encode = np.reshape(encode, encode.shape[1])
    return encode
'''


def beam_search_predictions(model, word2idx,idx2word,image, max_len,encoding_test,beam_index = 3):
    start = [word2idx["<start>"]]
    
    start_word = [[start, 0.0]]
    inception_model = InceptionV3(weights='imagenet')
    input = inception_model.input
    fc_layer = inception_model.layers[-2].output
    ft_extractor_model = Model(input, fc_layer)            
    
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = sequence.pad_sequences([s[0]], maxlen=max_len, padding='post')
            # e = encoding_test[image.split('/')[-1]]
            e = imgage_ft(image, ft_extractor_model)    

            preds = model.predict([np.array([e]), np.array(par_caps)])
            word_preds = np.argsort(preds[0])[-beam_index:]
            
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [idx2word[i] for i in start_word]

    final_caption = []
    
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    
    final_caption = ' '.join(final_caption[1:])
    return final_caption

def predict_captions(model, word2idx,idx2word,image,max_len, encoding_test):
    start_word = ["<start>"]
    inception_model = InceptionV3(weights='imagenet')
    input = inception_model.input
    fc_layer = inception_model.layers[-2].output
    ft_extractor_model = Model(input, fc_layer)
    while True:
        par_caps = [word2idx[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        # e = encoding_test[image.split('/')[-1]]
        e = imgage_ft(image, ft_extractor_model)
        
        preds = model.predict([np.array([e]), np.array(par_caps)])
        word_pred = idx2word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])


def eval(args):
    unique = pickle.load(open(os.path.join(args.pkl_save_path,'unique.p'), 'rb'))
    print('the length of unique is {}'.format(len(unique)))
    
    encoding_test = pickle.load(open(os.path.join(args.pkl_save_path,'encoded_images_test_inceptionV3.p'), 'rb'))
    
    word2idx = {val:index for index, val in enumerate(unique)}
    idx2word = {index:val for index, val in enumerate(unique)}
    
    vocab_size = len(unique)
    print('The vocab size is {}'.format(vocab_size))
    
    train_data = pd.read_csv(os.path.join(args.pkl_save_path, args.dataframe), delimiter = '\t')    
    image_ids = [i for i in train_data['image_id']]
    image_captions = [i for i in train_data['captions']]
    
    max_len = 0
    samples_per_epoch = 0
    for caption in image_captions:
        samples_per_epoch += len(caption.split())-1
        caption = caption.split()
        if len(caption) > max_len:
            max_len = len(caption)
    print('the max length of unique is {}'.format((max_len)))
    print('The samples per epoch is {}'.format(samples_per_epoch))
    
    print('Loading model from {}'.format(args.model_save_path))
    captioning_model = load_model(args.model_save_path)
    '''
    test_images_list = parse_data(args.test_images_file, args.image_folder_path)
    
    for i in range(0,10):
        print('-------------------------------------------------------------------------------------------------------------------')
        print ('Normal Max search:', predict_captions(model = captioning_model, word2idx = word2idx,idx2word = idx2word, image = test_images_list[i],max_len = max_len, encoding_test = encoding_test)) 
        print ('Beam Search, k=3:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=3))
        print ('Beam Search, k=5:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=5))
        print ('Beam Search, k=7:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=7))
        print('-------------------------------------------------------------------------------------------------------------------')
    
    '''
    test_images_list = [os.path.join(args.sample_test_images, i) for i in os.listdir(args.sample_test_images)]
    
    for i in range(0,len(test_images_list)):
        print('-------------------------------------------------------------------------------------------------------------------')
        print ('Normal Max search:', predict_captions(model = captioning_model, word2idx = word2idx,idx2word = idx2word, image = test_images_list[i],max_len = max_len, encoding_test = encoding_test)) 
        # print ('Beam Search, k=3:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=3))
        print ('Beam Search, k=5:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=5))
        # print ('Beam Search, k=7:', beam_search_predictions(model = captioning_model, word2idx = word2idx,idx2word = idx2word,image = test_images_list[i],max_len = max_len, encoding_test = encoding_test, beam_index=7))
        print('-------------------------------------------------------------------------------------------------------------------')
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', type=str, default='/home/sxg8458/keras_image_captioning/model/captioning_model.h5', help="Path to save pickle files")
    parser.add_argument('--image_folder_path', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flicker8k_Dataset/', help="Path to all the images")
    parser.add_argument('--test_images_file', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.testImages.txt', help="Path to test image file names")
    parser.add_argument('--pkl_save_path', type=str, default='/home/sxg8458/keras_image_captioning/pkl', help="Path to save pickle files")
    parser.add_argument('--dataframe', type=str, default='flickr8k_training_dataset.txt', help="Path to save pickle files")
    parser.add_argument('--sample_test_images', type=str, default='/home/sxg8458/keras_image_captioning/sample_test_data', help="Sample_Test_Files")
    args=parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    eval(args)
    
    