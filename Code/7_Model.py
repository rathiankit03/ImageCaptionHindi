# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 00:40:21 2019

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:11:49 2019

@author: Lenovo
"""
###  >>>>>>>>>>>>>>> Loading the Data <<<<<<<<<<<<<<<<<<<< ##############


from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
import time
import matplotlib.pyplot as plt


 
# load doc into memory
def LoadDoc(file_name):
	file = open(file_name, 'r', encoding = "utf-8")
	text = file.read()
	file.close()
	return text
 
# load a pre-defined list of photo identifiers
def LoadSet(file_name):
	document = LoadDoc(file_name)
	dataset = list()
	for line in document.split('\n'):
		if len(line) < 1:
			continue
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
 
# load clean descriptions into memory
def LoadCleanDescription(file_name, dataset):
	doc = LoadDoc(file_name)
	desc = dict()
	for line in doc.split('\n'):
		token = line.split()
		imageId, imageDesc = token[0], token[1:]
		if imageId in dataset:
			if imageId not in desc:
				desc[imageId] = list()
			descrption = 'startseq ' + ' '.join(imageDesc) + ' endseq'
			desc[imageId].append(descrption)
	return desc
 
# load photo features
def LoadPhotoFeatures(file_name, dataset):
	# load all features
	all_photo_features = load(open(file_name, 'rb'))
	# filter features
	features = {k: all_photo_features[k] for k in dataset}
	return features
 

# convert a dictionary of clean descriptions to a list of descriptions
def ToLines(desc):
	all_desc = list()
	for key in desc.keys():
		[all_desc.append(d) for d in desc[key]]
	return all_desc

# fit a tokenizer given caption descriptions
def CreateTokeizer(desc):
	l = ToLines(desc)
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(l)
	return tokenizer
 
# create sequences of images, input sequences and output words for an image
def CreateSequence(tokenizer, maxLength, desc, images, vocabSize):
	X1, X2, y = list(), list(), list()
	for key, desc_list in desc.items():
		for desc_sent in desc_list:
			sequence = tokenizer.texts_to_sequences([desc_sent])[0]
			for i in range(1, len(sequence)):
				in_sequence, out_sequence = sequence[:i], sequence[i]
				in_sequence = pad_sequences([in_sequence], maxlen=maxLength)[0]
				out_sequence = to_categorical([out_sequence], num_classes=vocabSize)[0]
				X1.append(images[key][0])
				X2.append(in_sequence)
				y.append(out_sequence)
	return array(X1), array(X2), array(y)


# Maximum length
def maxLength(descs):
	lines = ToLines(descs)
	return max(len(d.split()) for d in lines)

# define the captioning model
def DefineModel(vocabSize, maxLength):
    # feature extractor model
    input1 = Input(shape=(4096,))
    fe1 = Dropout(0.5, seed = 121)(input1)
    fe2 = Dense(128, activation='relu')(fe1)
    # sequence model
    input2 = Input(shape=(maxLength,))
    se1 = Embedding(vocabSize, 128, mask_zero=True)(input2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(128)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(128, activation='relu')(decoder1)
    outputs = Dense(vocabSize, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[input1, input2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    return model

            
# train dataset

# load training dataset (6K)
train = LoadSet('/project/ProjectData/Flickr_8k.trainImages.txt')
print('Training Dataset: %d' % len(train))
# descriptions
train_descriptions = LoadCleanDescription('/project/ProjectData/hindi1SentClean.txt', train)
print('Train Descriptions: train=%d' % len(train_descriptions))
# photo features
train_features = LoadPhotoFeatures('/project/ProjectData/Imagefeatures_backup.pkl', train)
print('Training Images: train=%d' % len(train_features))
# prepare tokenizer
tokenizer = CreateTokeizer(train_descriptions)
vocabSize = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocabSize)
# determine the maximum sequence length
maxLength = maxLength(train_descriptions)
print('Description Length: %d' % maxLength)
# prepare sequences
X1train, X2train, ytrain = CreateSequence(tokenizer, maxLength, train_descriptions, train_features, vocabSize)

#test dataset

# load test set
test = LoadSet('/project/ProjectData/Flickr_8k.testImages.txt')
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = LoadCleanDescription('/project/ProjectData/hindi1SentClean.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = LoadPhotoFeatures('/project/ProjectData/Imagefeatures_backup.pkl', test)
print('Photos: test=%d' % len(test_features))
# prepare sequences
X1test, X2test, ytest = CreateSequence(tokenizer, maxLength, test_descriptions, test_features, vocabSize)


# define the model
model = DefineModel(vocabSize, maxLength)
# define checkpoint callback
filename = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
start = time.time()
hist = model.fit([X1train, X2train], ytrain, epochs=10, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest)) 
end = time.time()
print("TIME TOOK {:3.2f}MIN".format((end - start )/60))
    
#Plot Graph

for label in ["loss","val_loss"]:
    plt.plot(hist.history[label],label=label)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
plt.savefig('HC1S_Ex7_ModelPerformance.png')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
