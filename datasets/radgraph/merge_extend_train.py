import json


with open('D:/studium/MIML/radgraph/radgraph/'+'detr_data_extend'+'.json', 'r') as f:
    data_extend = json.load(f)

with open('D:/studium/MIML/radgraph/radgraph/'+'detr_data_train'+'.json', 'r') as f:
    data_train = json.load(f)

data_extend.update(data_train)
print(len(data_extend))
with open('D:/studium/MIML/radgraph/radgraph/'+'detr_data_extend+train'+'.json', 'w') as outfile:
    json.dump(data_extend, outfile)