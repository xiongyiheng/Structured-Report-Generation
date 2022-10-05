"""
This file is to analyse the balance property between the classes of our generated dataset,
where the whole triple is treated as one class.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
None_obj = [25,18,24]
ls=[]
N = 15 # N largest values


with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/detr_SmartReporting_train.json", 'r') as f:
    ref_dict = json.load(f)

for key,ls_ls in ref_dict.items():
    for i in ls_ls:
        if i != None_obj:
            ls.append(i)

# print(len(ls))
# print(ls[0])
ls = np.asarray(ls)
unique, counts = np.unique(ls, return_counts=True,axis=0)
fre = counts/ls.shape[0]
#print(counts.shape)
# select the N largest values from the array
ind = np.argpartition(fre, -1*N)[-1*N:]  #return the index
selected_triple = unique[ind]

# mapping the numbers to the names
with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/num_words_mapping_original.json", 'r') as f:
    map_dict = json.load(f)
#print(map_dict['dis_ls'][2])
for i in range(len(selected_triple)):
    triple = selected_triple[i]
    #print(triple)
    #print(triple[0])
    #print(ind[i])
    print("["+map_dict['dis_ls'][triple[0]]+" "+map_dict['organ_ls'][triple[1]]+" "+map_dict['loc_ls'][triple[2]]+"] : "+str(counts[ind[i]]))


plt.bar(np.arange(unique.shape[0]),fre,0.2,label = 'Train_SmartReporting')
# plt.bar(np.arange(x0)+width,y1,width,label = 'Dev_'+mode)
plt.xlabel('Classes')
plt.ylabel('Frequency of occurrence')
plt.title('Occurrence frequency of in Train dataset')
#plt.xticks(np.arange(unique.shape[0]), labels=unique)
#plt.yticks(np.arange(0,0.02+max(np.max(fre)),step=0.02))
plt.legend(loc='best')
plt.show()