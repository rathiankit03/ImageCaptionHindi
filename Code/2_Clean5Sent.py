

### Hindi Text Data Cleaning and Processing: 

import numpy as np
import pandas as pd
from collections import Counter

# Loading Hindi text data:

hind_desc  = np.load('F:\ResearchProject\Code\Captioning\hindi_desc_final.npy', allow_pickle = True).item()


### Analysis 

#Distribution of word

# Dictonary to list:

lines = list()
for key, desc_list in hind_desc.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
            
def df_word(df_txt):
    vocabulary = []
    for txt in df_txt:
        vocabulary.extend(txt.split())
    print('Vocabulary Size: %d' % len(set(vocabulary)))
    ct = Counter(vocabulary)
    dfword = pd.DataFrame({"word":list(ct.keys()),"count":list(ct.values())})
    dfword = dfword.sort_values("count",ascending=False)
    dfword = dfword.reset_index()[["word","count"]]
    return(dfword)

dfword = df_word(lines)

# Top 50 words
Top50Word = dfword.head(50)
Top50Word.to_csv( 'F:/ResearchProject/Code/Captioning/OrignalWordDistribution.csv', sep=',', index=False, encoding = "utf-8-sig")

# Bottom 50 words 
Bottom50 = dfword.tail(50)
Bottom50.to_csv( 'F:/ResearchProject/Code/Captioning/LeastFrequentWord.csv', sep=',', index=False, encoding = "utf-8-sig")

### Cleaning the data

# Removing Number words from sentence 
    
def clean_descriptions(description):
    #stopwords = ['एक'   ]
    stopwords = ['एक' ]
    for key, desc_list in description.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            #tokenize 
            desc = desc.split()
            #Removing stop words
            desc = [word for word in desc if word not in stopwords]
            #store as string
            desc_list[i] = ' '.join(desc)
    return description
            
hind_desc = clean_descriptions(hind_desc)
        
# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# summarize vocabulary
vocabulary = to_vocabulary(hind_desc)
print('Vocabulary Size: %d' % len(vocabulary))


# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w', encoding="utf8")
	file.write(data)
	file.close()


# save to file
save_descriptions(hind_desc, 'F:\ResearchProject\Code\Captioning\hindiClean_desc.txt')