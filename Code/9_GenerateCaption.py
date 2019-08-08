from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

# extract features from each photo in the directory for VGG

def ExtractFeatures(file_name):
	model = VGG16()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	img = load_img(file_name, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	img = preprocess_input(img)
	img_feature = model.predict(img, verbose=0)
	return img_feature


# extract features from each photo in the directory for ResNet
"""
def ExtractFeatures(file_name):
	model = ResNet50()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	img = load_img(file_name, target_size=(224, 224))
	img = img_to_array(img)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	img = preprocess_input(img)
	img_feature = model.predict(img, verbose=0)
	return img_feature

# extract features from each photo in the directory for Ineception-V3

def ExtractFeatures(file_name):
	model = InceptionV3()
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	img = load_img(file_name, target_size=(229, 229))
	img = img_to_array(img)
	img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
	img = preprocess_input(img)
	img_feature = model.predict(img, verbose=0)
	return img_feature"""


# map an integer to a word
def WordForId(integer, tokenizer):
	for w, i in tokenizer.word_index.items():
		if i == integer:
			return w
	return None

# generate a description for an image
def GenerateDesc(model, tokenizer, image, maxLength):
	start_text = 'startseq'
	for i in range(maxLength):
		seq = tokenizer.texts_to_sequences([start_text])[0]
		seq = pad_sequences([seq], maxlen=maxLength)
		yword = model.predict([image,seq], verbose=0)
		yword = argmax(yword)
		word = WordForId(yword, tokenizer)
		if word is None:
			break
		start_text += ' ' + word
		if word == 'endseq':
			break
	return start_text

# load the tokenizer
tokenizer = load(open('F:/ResearchProject/Code/Captioning/tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 39
# load the model
model = load_model('F:/ResearchProject/Code/EncoderDecoder/model_19.h5')
# load and prepare the photograph
photo = ExtractFeatures('F:/ResearchProject/Dataset/Image Processing/Flicker8k_Dataset/133189853_811de6ab2a.jpg')
# generate description
description = GenerateDesc(model, tokenizer, photo, max_length)
print(description)


