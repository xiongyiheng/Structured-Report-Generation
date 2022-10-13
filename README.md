# Structured-Report-Generation
### Practical course in TUM department of Informatics: Machine Learning in Medical Imaging
Generating structured report dataset from scene graphs labels (RadGraph);
Investigating several neural models that as input will have the imaging data and as output should
present structured report.

### Installation 
We suggest using Conda to install and manage the environment. 

` conda env create -f environment.yml `

You can then rename the environment.

### Processing the datasets
In datasets/radgraph/ you can find all files related to process the dataset RadGraph.
More specifically,

`add_suggestive.py`: to add the "located in" between entities in "suggestive" relations
since in RadGraph it lacks such relations in these cases.

`dataset_analyse.py`: to analyse the balance property between all classes of RadGraph

`generate_data_transformer.py`: to generate dataset in json files for Transformer training

`generate_dataset.py`: to generate dataset in json files for baseline training

`without_OB_modified.py`: to generate the template based on training data with suffix `_train.json` or `_eval.json` and mapping `mapping_delete_meaningless.json`

`report_template.html`: to demonstrate how the structured report look like

`radgraph.py`: to load the data into the models(both for baseline and detr)

`detr_data_dev.json`: the val dataset for DETR with original 400+ studies

`detr_data_train.json`: the train dataset for DETR with original 400+ studies 

`detr_data_extend+train.json`: the train dataset for DETR with self-generated 2000+ studies. And for val, the origin one is used

`final_dataset_dev.json`: the val dataset for baseline

`final_dataset_train.json`: the train dataset for baseline



`mapping_original.json`: original mapping dictionary for all organ, loc and observations

`mapping_delete_meaningless.json`: the mapping dictionary deleting some meaningless items

### Model part
In /model/ dir there are some files to build the model. You could know their functions from their names ;)

### Train Model
`train_baseline.py`: to train the baseline in random split mode
`trian_baseline_k_fold`: to train the baseline in k fold mode
`train_detr.py`: to train the DETR model in random split mode
`train_k_fold`: to train the DETR model in k fold mode

After editing the dir of the datasets in script datasets/radgraph/radgraph.py, 
you may directly train the model with 
`CUDA_VISIBLE_DEVICES=7 python train_k_fold.py`
or
`CUDA_VISIBLE_DEVICES=7 python train_baseline_k_fold.py` for training the baseline.
