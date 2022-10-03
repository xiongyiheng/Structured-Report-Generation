import json
import numpy as np
with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/organ_loc_ob_smart_reporting.json", 'r') as f:
    ref_dict = json.load(f)

loc_ls = []
organ_ls = []
dis_ls = []
nr_cls = 0

for key,in_dict in ref_dict.items():
    organ_ls.append(key)
    for loc,ls in in_dict.items():
        nr_cls+=len(ls)
        dis_ls.extend(ls)
        loc_ls.append(loc)
#loc_ls = list(set(loc_ls))
# a_file = open("all Locations.json", "w")
# json.dump(loc_ls, a_file)
# a_file.close()
#print(loc_ls)
loc_ls = list(set(loc_ls))
organ_ls = list(set(organ_ls))
dis_ls = list(set(dis_ls))

#loc_ls.index('surface')
#loc_ls[103]

print(len(loc_ls))   #104   #80
print(len(organ_ls))  #87   #47
print(len(dis_ls))  #173    #126
#print(dis_ls[0])
print(nr_cls)

cls_ls = []
for key,in_dict in ref_dict.items():
    idx_organ = organ_ls.index(key)
    for loc,ls in in_dict.items():
        idx_loc = loc_ls.index(loc)
        for l in ls:
            idx_dis = dis_ls.index(l)
            cls_ls.append([idx_dis,idx_organ,idx_loc])

#print(cls_ls)
print(len(cls_ls))
#print(cls_ls.index([0, 10, 23]))
#cls_array = np.array(cls_ls)

num_words_mapping = {}
num_words_mapping['cls_ls'] = cls_ls
num_words_mapping['dis_ls'] = dis_ls
num_words_mapping['organ_ls'] = organ_ls
num_words_mapping['loc_ls'] = loc_ls

with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/num_words_mapping_original.json", "w") as outfile:
    json.dump(num_words_mapping, outfile)
