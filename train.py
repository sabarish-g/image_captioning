import numpy as np
import argparse
import pdb
import pandas as pd
import pickle
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.models import Model


def create_model(max_len, vocab_size,embedding_size = 300):
    image_model = Sequential([Dense(300, input_shape=(2048,), activation='relu'),RepeatVector(max_len)])
    caption_model = Sequential([Embedding(vocab_size, embedding_size, input_length=max_len),LSTM(256, return_sequences=True),TimeDistributed(Dense(300))])

    final_model = Concatenate()([image_model.output, caption_model.output])
    final_model = Bidirectional(LSTM(256, return_sequences=False))(final_model)
    final_model = Dense(vocab_size)(final_model)
    final_model = Activation('softmax')(final_model)
    total_model = Model([image_model.input, caption_model.input], final_model)
    total_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    return total_model


def data_generator(encoding_train,word2idx,vocab_size,max_len,batch_size=32):
    partial_caps = []
    next_words = []
    images = []
    df = pd.read_csv(os.path.join(args.pkl_save_path, args.dataframe), delimiter='\t')
    df = df.sample(frac=1) 
    iter = df.iterrows()

    captions = []
    imgs = []

    for i in range(0,df.shape[0]):
        x = next(iter)
        captions.append(x[1][1])
        imgs.append(x[1][0])
    
    count = 0
    while True:
        
        for idx, text in enumerate(captions):
            current_image = encoding_train[imgs[idx]]
            for i in range(len(text.split()) -1):
                count = count + 1
                partial = [word2idx[txt] for txt in text.split()[:i+1]]
                partial_caps.append(partial)
                # Initializing with zeros to create a one-hot encoding matrix
                # This is what we have to predict
                # Hence initializing it with vocab_size length
                n = np.zeros(vocab_size)
                # Setting the next word to 1 in the one-hot encoded matrix
                n[word2idx[text.split()[i+1]]] = 1
                next_words.append(n)  
                images.append(current_image)

                if count>=batch_size:
                    # pdb.set_trace()

                    next_words = np.asarray(next_words)
                    images = np.asarray(images)
                    
                    partial_caps = sequence.pad_sequences(partial_caps, maxlen=max_len, padding='post')
                    
                    yield [[images, partial_caps], next_words]
                    partial_caps = []
                    next_words = []
                    images = []
                    count = 0
                


def train(args):
    encoding_train = pickle.load(open(os.path.join(args.pkl_save_path,'encoded_images_train_inceptionV3.p'), 'rb'))
    # encoding_val = pickle.load(open(os.path.join(pkl_save_path,'encoded_images_val_inceptionV3.p'), 'rb'))
    # encoding_test = pickle.load(open(os.path.join(pkl_save_path,'encoded_images_test_inceptionV3.p'), 'rb'))

    unique = pickle.load(open(os.path.join(args.pkl_save_path,'unique.p'), 'rb'))
    print('the length of unique is {}'.format(len(unique)))
    
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
    
    captioning_model = create_model(max_len = max_len, vocab_size = vocab_size,embedding_size = 300)
    print(captioning_model.summary())
    captioning_model.fit_generator(data_generator(encoding_train = encoding_train,word2idx = word2idx,vocab_size = vocab_size,max_len = max_len,batch_size=args.batch_size), \
                                    samples_per_epoch=np.ceil(samples_per_epoch/args.batch_size),epochs = 20,verbose=1)
    
    captioning_model.save(os.path.join(args.model_save_path,"captioning_model.h5"))
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pkl_save_path', type=str, default='/home/sxg8458/keras_image_captioning/pkl', help="Path to save pickle files")
    parser.add_argument('--dataframe', type=str, default='flickr8k_training_dataset.txt', help="Path to save pickle files")
    parser.add_argument('--model_save_path', type=str, default='/home/sxg8458/keras_image_captioning/model', help="Path to save pickle files")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size")
    
    args=parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    train(args)