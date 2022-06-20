########## This file is to add "located_at" for the OBS that
#####   are lack of "located_at" AND be "suggestive_of" by other OBS.
import json

def write_json(data,file_path):#'D:/studium/MIML/radgraph/radgraph/train_add_sug.json'
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)



# with open('D:/studium/MIML/radgraph/radgraph/train.json', 'r') as f:
#     data = json.load(f)

# for key in data.keys():  # key : "p18/p18004941/s58821758.txt"
#     new_dict = data[key]
#     entities = new_dict['entities']
#     for new_key in entities.keys():
#         entity = entities[new_key]     #entity can be "6":{}      "cancer"
#         relations = entity['relations']
#         contain_sugg = False
#         contain_loca = False
#         contain_modify = False
#         ind_located = None
#         for i in range(len(relations)):
#             if relations[i][0] == "suggestive_of":
#                 contain_sugg= True
#             if relations[i][0] == "located_at":
#                 contain_loca = True
#                 ind_located = i     ### store the ind of "located_at" relation
#             if relations[i][0] == "modify":
#                 contain_modify = True
#         if contain_sugg and contain_loca:
#             for j in range(len(relations)):
#                 if relations[j][0] == "suggestive_of":
#                     relations2 = entities[relations[j][1]]['relations']
#                     contain_loca2 = False
#                     for k in range(len(relations2)):
#                         if relations2[k][0] == "located_at":
#                             contain_loca2 = True
#                     if not contain_loca2:
#                         entities[relations[j][1]]['relations'].append(entity['relations'][ind_located])
#                     else:
#                         pass
#         if contain_sugg and (not contain_loca) and (not contain_modify):
#             for j in range(len(relations)):
#                 if relations[j][0] == "suggestive_of":
#                     relations2 = entities[relations[j][1]]['relations']
#                     contain_loca2 = False
#                     for k in range(len(relations2)):
#                         if relations2[k][0] == "located_at":
#                             contain_loca2 = True
#                     if contain_loca2:
#                         entity['relations'].append(entities[relations[j][1]]['relations'])
                        #entities[relations[j][1]]['relations'].append(entity['relations'][ind_located])

#write_json(data,'D:/studium/MIML/radgraph/radgraph/train_add_sug.json')


with open('D:/studium/MIML/radgraph/radgraph/train_add_sug.json', 'r') as f:
    data = json.load(f)

for key in data.keys():  # key : "p18/p18004941/s58821758.txt"
    new_dict = data[key]
    entities = new_dict['entities']
    for new_key in entities.keys():
        entity = entities[new_key]     #entity can be "6":{}      "cancer"
        relations = entity['relations']
        contain_sugg = False
        contain_loca = False
        contain_modify = False
        ind_located = None
        for i in range(len(relations)):
            if relations[i][0] == "suggestive_of":
                contain_sugg= True
            if relations[i][0] == "located_at":
                contain_loca = True
                ind_located = i     ### store the ind of "located_at" relation
            if relations[i][0] == "modify":
                contain_modify = True
        if (not contain_loca) and (contain_sugg) and (not contain_modify) :
            print("fuck")