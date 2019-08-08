### Hindi Text Data Cleaning and Processing: 

import numpy as np

# Loading Hindi text data:

hind_desc  = np.load('F:\ResearchProject\Code\Captioning\hindi_desc_final.npy', allow_pickle = True).item()


# Taking first caption from five captions

hindi_1_desc = dict()
for key, value in hind_desc.items():
    for i in range(1):
        hindi_1_desc.update({key : value[i]})

#Converting the dictonary to list
        
SingleLines = list()
for key, desc_list in hindi_1_desc.items():
    SingleLines.append(key + ' ' + desc_list)
    
    
# save descriptions to file, one per line
def save_descriptions_1sentence(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        lines.append(key + ' ' + desc_list)
    data = '\n'.join(lines)
    file = open(filename, 'w', encoding="utf8")
    file.write(data)
    file.close()
    
#save to file
save_descriptions_1sentence(hindi_1_desc, 'F:/ResearchProject/Code/EncoderDecoder/1Sentence/hindi1Sent.txt')