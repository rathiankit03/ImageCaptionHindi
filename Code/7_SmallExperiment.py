
from numpy import array
from numpy import argmax
from pandas import DataFrame
from nltk.translate.bleu_score import corpus_bleu
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers.merge import add
from keras.layers import Dropout


# load doc into memory
def LoadDoc(file_name):
	file = open(file_name, 'r', encoding = "utf-8")
	text_data = file.read()
	file.close()
	return text_data

# load a pre-defined list of photo identifiers
def LoadSet(file_name):
	document = LoadDoc(file_name)
	dataset = list()
	for l in document.split('\n'):
		if len(l) < 1:
			continue
		identifier = l.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# split a dataset into train/test elements
def TrainTestSplit(dataset):
	order= sorted(dataset)
	return set(order[:101]), set(order[100:200])

# load clean descriptions into memory
def LoadCleanDescription(file_name, dataset):
	document = LoadDoc(file_name)
	descriptions = dict()
	for l in document.split('\n'):
		tokens = l.split()
		imageId, imageDesc = tokens[0], tokens[1:]
		if imageId in dataset:
			# store
			descriptions[imageId] = 'startseq ' + ' '.join(imageDesc) + ' endseq'
	return descriptions

# load photo features
def LoadPhotoFeatures(file_name, dataset):
	AllFeatures = load(open(file_name, 'rb'))
	features = {k: AllFeatures[k] for k in dataset}
	return features

# fit a tokenizer given caption descriptions
def CreateTokenizer(descriptions):
	l = list(descriptions.values())
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(l)
	return tokenizer

# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, description, photo, maxLength):
	Ximages, XSeq, y = list(), list(),list()
	vocabSize = len(tokenizer.word_index) + 1
	sequence = tokenizer.texts_to_sequences([description])[0]
	for i in range(1, len(sequence)):
		in_sequence, out_sequence = sequence[:i], sequence[i]
		in_sequence = pad_sequences([in_sequence], maxlen=maxLength)[0]
		out_sequence = to_categorical([out_sequence], num_classes=vocabSize)[0]
		Ximages.append(photo)
		XSeq.append(in_sequence)
		y.append(out_sequence)
	return [Ximages, XSeq, y]

# define the captioning model
def DefineModel(vocabSize, maxLength):
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5, seed = 121)(inputs1)
    fe2 = Dense(128, activation='relu')(fe1)
    fe3 = RepeatVector(max_length)(fe2)
    
    # embedding
    inputs2 = Input(shape=(max_length,))
    emb2 = Embedding(vocab_size, 50, mask_zero=True)(inputs2)
    emb2a = Dropout(0.5, seed = 121)(emb2)
    emb3 = LSTM(128, return_sequences=True)(emb2a)
    emb4 = LSTM(128, return_sequences=True)(emb3)
    emb5 = TimeDistributed(Dense(128, activation='relu'))(emb4)
    # merge inputs
    merged = add([fe3, emb5])
    # language model (decoder)
    lm1 = Dropout(0.5, seed = 121)(merged)
    lm2 = LSTM(500)(lm1)
    lm3 = Dense(500, activation='relu')(lm2)
    outputs = Dense(vocab_size, activation='softmax')(lm3)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer = 'adam' , metrics=['accuracy'])
    print(model.summary())
    plot_model(model, show_shapes=True, to_file='plot.png')
    return model


# data generator, intended to be used in a call to model.fit_generator()
def DataGenerator(descriptions, ImageFeatures, tokenizer, maxLength, nStep):
	while 1:
		keys = list(descriptions.keys())
		for i in range(0, len(keys), nStep):
			Ximages, XSequence, y = list(), list(),list()
			for j in range(i, min(len(keys), i+nStep)):
				imageId = keys[j]
				photo = ImageFeatures[imageId][0]
				desc = descriptions[imageId]
				input_image, input_sequence, output_word = create_sequences(tokenizer, desc, photo, max_length)
				for k in range(len(input_image)):
					Ximages.append(input_image[k])
					XSequence.append(input_sequence[k])
					y.append(output_word[k])
			yield [[array(Ximages), array(XSequence)], array(y)]

# map an integer to a word
def WordForId(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def GenerateDesc(model, tokenizer, image, maxLength):
	input_text = 'startseq'
	for i in range(maxLength):
		seq = tokenizer.texts_to_sequences([input_text])[0]
		seq = pad_sequences([seq], maxlen=maxLength)
		yhat = model.predict([image,seq], verbose=0)
		yhat = argmax(yhat)
		word = WordForId(yhat, tokenizer)
		if word is None:
			break
		input_text += ' ' + word
		if word == 'endseq':
			break
	return input_text

# evaluate the skill of the model
def evaluate_model(model, descriptions, image, tokenizer, maxLength):
	y, yhat = list(), list()
	for key, value in descriptions.items():
		predict = GenerateDesc(model, tokenizer, image[key], maxLength)
		y.append([value.split()])
		yhat.append(predict.split())
	# calculate BLEU score
	bleu = corpus_bleu(y, yhat)
	return bleu

# load dev set
dataset = LoadSet('/project/ProjectData/Flickr_8k.devImages.txt')
print('Length of Dataset: %d' % len(dataset))
# train-test split
train, test = TrainTestSplit(dataset)
# descriptions
train_descriptions = LoadCleanDescription('/project/ProjectData/hindi1SentClean.txt', train)
test_descriptions = LoadCleanDescription('/project/ProjectData/hindi1SentClean.txt', test)
print('Descriptions: train=%d, test=%d' % (len(train_descriptions), len(test_descriptions)))
# photo features
train_features = LoadPhotoFeatures('/project/ProjectData/ResNet50features.pkl', train)
test_features = LoadPhotoFeatures('/project/ProjectData/ResNet50features.pkl', test)
print('Photos: train=%d, test=%d' % (len(train_features), len(test_features)))
# prepare tokenizer
tokenizer = CreateTokenizer(train_descriptions)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# determine the maximum sequence length
max_length = max(len(s.split()) for s in list(train_descriptions.values()))
print('Description Length: %d' % max_length)

# define experiment
model_name = 'Ex18'
verbose = 2
n_epochs = 30
n_photos_per_update = 2
n_batches_per_epoch = int(len(train) / n_photos_per_update)
n_repeats = 5

# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
	model = DefineModel(vocab_size, max_length)
	model.fit_generator(DataGenerator(train_descriptions, train_features, tokenizer, max_length, n_photos_per_update), steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose)
	# evaluate model on training data
	train_score = evaluate_model(model, train_descriptions, train_features, tokenizer, max_length)
	test_score = evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
	train_results.append(train_score)
	test_results.append(test_score)
	print('>%d: train=%f test=%f' % ((i+1), train_score, test_score))
    
# save results to file
df = DataFrame()
df['train'] = train_results
df['test'] = test_results
print(df.describe())
df.to_csv(model_name+'.csv', index=False)
