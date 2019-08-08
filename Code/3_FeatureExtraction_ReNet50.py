from os import listdir
from pickle import dump
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model




# Image Extraction using ResNet50: 

# extract features 

def features_extraction(path):
	# loading the ReNet50 pre-trained model
	model = ResNet50()
	# re-structuring the model, removing last layer from the model
	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
	print(model.summary())
	# extracting features from each photo
	features = dict()
	for word in listdir(path):
		# loading a single image from file
		file = path + '/' + word
		img = load_img(file, target_size=(229, 229))
		# convering the image pixel to array
		img = img_to_array(img)
		# reshape
		img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
		# prepare the image for the ReNet50 model
		img = preprocess_input(img)
		# get features
		feature = model.predict(img, verbose=0)
		# get image id
		img_id = word.split('.')[0]
		# store feature
		features[img_id] = feature
		print('>%s' % word)
	return features
 
# extract features from all images
path = 'F:\ResearchProject\Dataset\ImageProcessing\Flicker8k_Dataset'
features = features_extraction(path)
print('Extracted VGG Features: %d' % len(features))
# save to file
dump(features, open('ReNet50features.pkl', 'wb'))