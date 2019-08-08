#### Machin Translation English to Hindi using Google API
import os
from google.cloud import translate
import numpy as np
import ast 

# loading a file 

def load_file(fileName):
	file = open(fileName, 'r')
	text = file.read()
	file.close()
	return text

doc = load_file("F:\\ResearchProject\\Dataset\\ImageProcessing\\Flickr8k.token.txt")

# Loading English Description 

def load_descriptions(doc):
	mapping = dict()
	for line in doc.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		image_id, image_desc = tokens[0], tokens[1:]
		image_id = image_id.split('.')[0]
		image_desc = ' '.join(image_desc)
		if image_id not in mapping:
			mapping[image_id] = list()
		mapping[image_id].append(image_desc)
	return mapping

# English Description
descriptions = load_descriptions('F:/ResearchProject/Code/ProjectData/hindi_desc.txt')

## Calling Google API

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'F:\ResearchProject\Google Translation-1d212fe2eb3f.json'


## Translation from English to Hindi


hindi_desc = dict()

for i in range(487, len(descriptions)):
    translate_client = translate.Client()
    text = list(descriptions.values())[i]
    hinditext = []
    for j in range(0, len(text)):
        hinditext.append(format(translate_client.translate(text[j], target_language = 'hi' )))
    hindi_desc.update({list(descriptions.keys())[i] : hinditext})
    
list(hindi_desc.values())[8090]
    
hindi_desc_1_486 = hindi_desc

np.save('hindi_1_486.npy', hindi_desc_1_486)

## Removing other text and kepping only hindi text

hindi_desc_final = {}

for i in range(0, len(hindi_desc)):
    imageList = list(hindi_desc.values())[i]
    hindi_text_list = []
    for j in range(0, len(imageList)):
        stringToKey = ast.literal_eval(imageList[j])
        hindi_text_list.append(stringToKey.get('translatedText'))
    hindi_desc_final.update({list(hindi_desc.keys())[i] : hindi_text_list})

np.save('hindi_desc_final.npy', hindi_desc_final) #Final

    
# save Hindi description

def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding="utf8")
	file.write(data)
	file.close()


# save Hindi Caption Dataset
    
save_descriptions(hind_desc, 'F:\ResearchProject\Code\Captioning\hindi_desc.txt')
