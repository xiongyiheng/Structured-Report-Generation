import json
import os
from collections import Counter

# p = os.getcwd().
def mapping_name(dict,value):
    for key,ls in dict.items():
        if value in ls:
            return key
    # if cant find the key
    return None


mapping = {"lung":["lung","pulmonary"],
           "pleural":["pleural","plerual"],
           "heart":["heart","cardiac","retrocardiac"],
           "mediastinum":["mediastinal","cardiomediastinal","mediastinum"],
           "lobe":["lobe","lobar"],
           "hilar":["hilar","hila","hilus","perihilar"],
           "vascular":["vascular","vasculature","bronchovascular","venous","aortic","vein","arteries","vasculatiry","aorta","artery","vessel"],
           "chest":["chest","thorax"],
           "cardiopulmonary":["cardiopulmonary"],
           "basilar":["bibasilar","basilar","base","bibasal","basal"],
           "diaphragm":["hemidiaphragm","diaphragm"],
           "rib":["rib"],
           "stomach":["stomach"],
           "spine":["spine"],
            }

with open('D:/studium/MIML/radgraph/radgraph/train.json', 'r') as f:
    data = json.load(f)
organs = []

###extract the labels###
for key in data.keys():
    new_dict = data[key]
    entities = new_dict['entities']
    for new_key in entities.keys():
        entity = entities[new_key]
        if entity['label'] == 'ANAT-DP':
            relations = entity['relations']
            is_contain_modify = False
            for i in range(len(relations)):
                if relations[i][0] == 'modify':
                    is_contain_modify = True
            if is_contain_modify == False:
                if entity['tokens'][-1] == 's' and entity['tokens'][-3:-1] != "ou" :
                    entity['tokens'] = entity['tokens'][:-1]
                entity['tokens'] = mapping_name(mapping,entity['tokens'])
                if entity['tokens'] != None:
                    organs.append(entity['tokens'].lower())

            #if len(relations) == 0:
            #    for i in range(len(relations)):
            #        if relations[i][0] == 'modify':
            #            idx = relations[i][1]
            #            if entities[idx]['label'] == 'OBS-DA':
            #                print(1)



### statistic and post-process###
result = Counter(organs)
sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
print(sorted_result)


#output distince outputs
organs = list(set(organs))

### print the result###
# print(len(organs))
# print(organs)