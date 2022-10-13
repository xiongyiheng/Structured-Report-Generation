## Tutorial
### How to anlayse the fre of each cls?
The only three relevant files are smart_reporting/num_words_mapping_origin.json, dataset_analyse_whole.py and detr_SmartReporting_train.json.

Edit the parameter in file dataset_analyse_whole.py in line 11 to the number of classes you want to print out. 
```
N = 5
```
And also the file dir of the two json files.
```
with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/detr_SmartReporting_train.json", 'r') as f:
    ref_dict = json.load(f)
```
```
with open("D:/studium/MIML/radgraph/radgraph/smart_reporting/num_words_mapping_original.json", 'r') as f:
    map_dict = json.load(f)
```

Then run the py file ``` python dataset_analyse_whole.py``` to print the counts of the classes out and also a frequency bar diagram.
