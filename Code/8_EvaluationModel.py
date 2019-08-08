

############## >>>>>>>>>>>> Evaluating Model <<<<<<<<<<<<<<<<<<<<<##############################

from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


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

# Maximum length
def maxLength(descs):
	lines = ToLines(descs)
	return max(len(d.split()) for d in lines)

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

# evaluate the skill of the model
def EvaluateModel(model, descs, images, tokenizer, maxLength):
	y, yhat = list(), list()
	for key, desc_list in descs.items():
		yword = GenerateDesc(model, tokenizer, images[key], maxLength)
		references = [d.split() for d in desc_list]
		y.append(references)
		yhat.append(yword.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(y, yhat, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(y, yhat, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(y, yhat, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(y, yhat, weights=(0.25, 0.25, 0.25, 0.25)))
    
### prepare test set

# Load train set  
train = LoadSet('F:/ResearchProject/Dataset/Image Processing/Flickr_8k.trainImages.txt')
# load test set
test = LoadSet('F:/ResearchProject/Dataset/Image Processing/Flickr_8k.testImages.txt')
# descriptions
test_descriptions = LoadCleanDescription('F:/ResearchProject/Code/Captioning/hindi_desc.txt', test)
# photo features
test_features = LoadPhotoFeatures('F:/ResearchProject/Code/Imagefeatures_backup.pkl', test)
# descriptions
train_descriptions = LoadCleanDescription('F:/ResearchProject/Code/Captioning/hindi_desc.txt', train)
# prepare tokenizer
tokenizer = CreateTokeizer(train_descriptions)
vocabSize = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocabSize)
# determine the maximum sequence length
maxLength = maxLength(train_descriptions)
print('Description Length: %d' % maxLength)


# load the model
model = load_model('F:/ResearchProject/Code/model_10.h5')
# evaluate model
EvaluateModel(model, test_descriptions, test_features, tokenizer, maxLength)



