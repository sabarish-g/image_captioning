# Image Captioning using CNN and LSTMs
An implementation of image captioning.

The model uses the Flickr8k dataset. The training set has 6k images and 5 captions for every image. The test and validation have 1k images with 5 captions each. The dataset can be downloaded at https://forms.illinois.edu/sec/1713398 .


First complete preprocessing to generate the tokens.
```
python preprocessing.py 
```
For training and testing: 
The train file will save model in the directory as well.
```
python train.py
python eval.py
```

Sample results on random images from google

![Screenshot](caption_result.PNG)
