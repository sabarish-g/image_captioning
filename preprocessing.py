import numpy as np
import os
import argparse
import pickle
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
import pandas as pd
from tqdm import tqdm
from PIL import Image
from keras.models import Model

import pdb

def create_dict(captions_data):
    data_dict = {}
    for idx, value in enumerate(captions_data):
        # pdb.set_trace()
        value = value.split('\t')
        value[0] = value[0].split('#')[0]
        if value[0] in data_dict:
            data_dict[value[0]].append(value[1])
        else:
            data_dict[value[0]] = [value[1]]
    return data_dict

def parse_data(file,image_folder_path):
    with open(file, 'r') as f:
        images_list = f.read().strip().split('\n')
    # pdb.set_trace()
    images_list = [os.path.join(image_folder_path, i) for i in images_list]
    print('Total files processed is {}'.format(len(images_list)))
    return images_list


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

def encode_images(image_list, pkl_save_path, model, phase = 'train'):
    encoded_images = {}
    for i in range(0,len(image_list)):
        if i%50 == 0 and i!=0 : print('images processed {} out of {}'.format(i,len(image_list)))
        encoded_images[(image_list[i].split('/')[-1])] = imgage_ft(image_list[i],model)
    if phase == 'train':
        with open(os.path.join(pkl_save_path,'encoded_images_train_inceptionV3.p'), 'wb') as encoded_pickle:
            pickle.dump(encoded_images, encoded_pickle)
    elif phase == 'val':
        with open(os.path.join(pkl_save_path,'encoded_images_val_inceptionV3.p'), 'wb') as encoded_pickle:
            pickle.dump(encoded_images, encoded_pickle)
    elif phase == 'test':
        with open(os.path.join(pkl_save_path,'encoded_images_test_inceptionV3.p'), 'wb') as encoded_pickle:
            pickle.dump(encoded_images, encoded_pickle)



def preprocessing(args):
    captions_file = open(args.token, 'r').read().strip().split('\n')
    dict_data = create_dict(captions_file)
    
    train_images_list = parse_data(args.train_images_file, args.image_folder_path)
    val_images_list = parse_data(args.val_images_file, args.image_folder_path)
    test_images_list = parse_data(args.test_images_file, args.image_folder_path)
    # pdb.set_trace()
    # inception_model = InceptionV3(weights='imagenet')
    # input = inception_model.input
    # fc_layer = inception_model.layers[-2].output
    # ft_extractor_model = Model(input, fc_layer)
    # encode_images(train_images_list, pkl_save_path = args.pkl_save_path, phase ='train', model = ft_extractor_model)
    # encode_images(val_images_list, pkl_save_path = args.pkl_save_path, phase ='val', model = ft_extractor_model)
    # encode_images(test_images_list, pkl_save_path = args.pkl_save_path, phase ='test', model = ft_extractor_model)
    
    train_d = {}
    for i in range(0,len(train_images_list)):
        image_name = train_images_list[i].split('/')[-1]
        if image_name in dict_data:
            train_d[image_name] = dict_data[image_name]
    
    captions = []
    for k,v in train_d.items():
        for caption in v:
            captions.append('<start> '+caption+' <end>')

    words = [i.split() for i in captions]

    unique = []
    for i in words:
        unique.extend(i)
    unique = list(set(unique))

    with open(os.path.join(args.pkl_save_path,'unique.p'), 'wb') as pickle_d:
        pickle.dump(unique, pickle_d)
    
    f = open(os.path.join(args.pkl_save_path,'flickr8k_training_dataset.txt'), 'w')
    f.write("image_id\tcaptions\n")

    for key, val in train_d.items():
        for i in val:
            f.write(key + "\t" + "<start> " + i +" <end>" + "\n")
    f.close()
    
    

    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flickr8k.lemma.token.txt', help="Path to captions file")
    parser.add_argument('--train_images_file', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.trainImages.txt', help="Path to train image file names")
    parser.add_argument('--val_images_file', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.devImages.txt', help="Path to val image file names")
    parser.add_argument('--test_images_file', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flickr_8k.testImages.txt', help="Path to test image file names")
    parser.add_argument('--pkl_save_path', type=str, default='/home/sxg8458/keras_image_captioning/pkl', help="Path to save pickle files")
    parser.add_argument('--image_folder_path', type=str, default='/shared/kgcoe-research/mil/Flickr8k/Flicker8k_Dataset/', help="Path to all the images")
    args=parser.parse_args()
    print '--------------------------------'
    for key, value in vars(args).items():
        print key, ' : ', value
    print '--------------------------------'
    preprocessing(args)