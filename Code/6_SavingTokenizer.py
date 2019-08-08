# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 00:02:43 2019

@author: Lenovo
"""

######### Saving Tokenizer ##############

from keras.preprocessing.text import Tokenizer
from pickle import dump

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


train = LoadSet('F:/ResearchProject/Dataset/ImageProcessing/Flickr_8k.trainImages.txt')
print('Train Dataset: %d' % len(train))
# descriptions
train_descriptions = LoadCleanDescription('F:/ResearchProject/Code/English/descriptions1SentUnClean.txt', train)
print(' Train Descriptions: train=%d' % len(train_descriptions))
# prepare tokenizer
tokenizer = CreateTokeizer(train_descriptions)
# save the tokenizer
dump(tokenizer, open('F:/ResearchProject/Code/ProjectData/EnglishUnclean1SentTokenizer.pkl', 'wb'))