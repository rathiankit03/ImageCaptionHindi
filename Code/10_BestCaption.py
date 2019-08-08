# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:46:55 2019

@author: Lenovo
"""

from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import pickle
from nltk.translate.bleu_score import corpus_bleu



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
tokenizer = load(open('F:/ResearchProject/Code/ProjectData/HindiCleanTokenizer.pkl', 'rb'))
tokenizer
# pre-define the max sequence length (from training)
maxLength = 39 #Hindi - 39, Hindi 1 Sent - 34,     EnglishClean - 34, English UnClean - 38, Hindi 1 Sent Clean - 33, English1SentClean
# load the model
model = load_model('F:/ResearchProject/Code/EncoderDecoder/OneRun/HindiClean.h5')
    
### Predicted Caption ####

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

# load photo features
def LoadPhotoFeatures(file_name, dataset):
	# load all features
	all_photo_features = load(open(file_name, 'rb'))
	# filter features
	features = {k: all_photo_features[k] for k in dataset}
	return features

# load test set

test = LoadSet('F:/ResearchProject/Code/ProjectData/Flickr_8k.testImages.txt')
print('Test Dataset: %d' % len(test))

# load the train set

train = LoadSet('F:/ResearchProject/Code/ProjectData/Flickr_8k.trainImages.txt')
print('Train Dataset: %d' % len(train))

#Load Dev set

dev = LoadSet('F:/ResearchProject/Code/ProjectData/Flickr_8k.devImages.txt')
print('Development Dataset: %d' % len(dev))

# Total

img = test | train 
print('Total Dataset: %d' % len(img))

#Load Test photo

test_features = LoadPhotoFeatures('F:/ResearchProject/Code/ProjectData/Imagefeatures_backup.pkl', img)

# generate description
PredCap = dict()
for key,value in test_features.items():
    photo = value
    desc = GenerateDesc(model, tokenizer, photo, maxLength)
    PredCap.update({key:desc})
 
#Save PredCap in pickle file    
file = open("F:/ResearchProject/Code/PredCap/HindiClean.pkl","wb")
pickle.dump(PredCap,file)
file.close()

#Creating PredCap Dataframe
PredCapDf = pd.DataFrame(list(PredCap.items()), columns=['Image', 'PredCap'])

# Loading PredCap from the system
#PredCap = load(open("F:/ResearchProject/Code/Hindi1SentCleanExp/Ex7_pred.pkl", 'rb'))

#Saving one description 

def save_descriptions_1sentence(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        lines.append(key + ' ' + desc_list)
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding="utf8")
    file.write(data)
    file.close()
    
save_descriptions_1sentence(PredCap, 'F:/ResearchProject/Code/PredCap/UnCleanHindi1Sent.txt')

"""def load_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = '' + ' '.join(image_desc) + ''
			# store
			descriptions[image_id].append(desc)
	return descriptions

a = load_descriptions('F:/ResearchProject/Code/BestCaption/PredCapHindi1SentClean.txt', img)"""

## Remove startseq and endseq from the sentences

# Removed startseq
for key, value in PredCap.items():
    sent = value
    sent = sent.replace('startseq', '')
    PredCap.update({key:sent})
    
# Removed endseq

for key, value in PredCap.items():
    sent = value
    sent = sent.replace('endseq', '')
    PredCap.update({key:sent})

    
#### True Caption ####

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

# descriptions
TrueCap = LoadCleanDescription('F:/ResearchProject/Code/ProjectData/hindi1SentClean.txt', img)
print('TrueCap Descriptions: train=%d' % len(TrueCap))



# Evaluation - BLEU


#Refernce
refernces = list()
for key, value in TrueCap.items():
    value = [word.split() for word in value]
    refernces.append(value)

r = list()    
for i in range(len(refernces)):
   r.append([refernces[i]])


    
#Candidate
cand = list()
for key, value in PredCap.items():
    cand.append(''.join(list(value)))

candidate = list()
for i in range(len(cand)): 
    splitword = cand[i].split()
    candidate.append(splitword)
    
c = list()    
for i in range(len(candidate)):
   c.append([candidate[i]])



# BLEU - n gram

Bleu = list()
for i in range(0,7000):
    score = corpus_bleu(r[i], c[i], weights=(0.25, 0.25, 0.25, 0.25),     ) ## Bleu-4
    Bleu.append(score)

ImageID = list(PredCap.keys())

Bleu1Df = pd.DataFrame({"Bleu-1": Bleu, "Image" : ImageID})
Bleu1Df = Bleu1Df.sort_values(["Bleu-1","Image"], ascending = [False, False] )

#### BestCaption 

BestCaptionDf = Bleu1Df[Bleu1Df["Bleu-1"] > 0.3]

df = pd.merge(BestCaptionDf, PredCapDf, on = "Image" )

# Saving the best caption in CSV file
df.to_csv( 'F:/ResearchProject/Code/Hindi1SentCleanExp/Ex7_BL2.csv', sep=',', index=False, encoding = "utf-8-sig")

