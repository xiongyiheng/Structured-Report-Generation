"""
This file is to analyse the balance property between all classes of RadGraph.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

nr_cls = [31, 25, 29]#[126,47,80]

counts_dis = np.zeros(shape=[2,nr_cls[0]])
counts_organ = np.zeros(shape=[2,nr_cls[1]])
counts_loc = np.zeros(shape=[2,nr_cls[2]])



##############train#############
with open("D:/studium/MIML/radgraph/radgraph/premium_selected/detr_data_extend_selected_20.json", 'r') as f:
    ref_dict = json.load(f)

#126, 47, 80
ds_ls = []
organ_ls = []
loc_ls = []
for key,ls_ls in ref_dict.items():
    ls = np.asarray(ls_ls)
    ds_ls.extend(ls[:,0])
    organ_ls.extend(ls[:,1])
    loc_ls.extend(ls[:,2])

ds_array = np.asarray(ds_ls)
#print(ds_array)
ds_array = np.delete(ds_array, np.where(ds_array == nr_cls[0]))
#print(ds_array)
print(ds_array.shape)

organ_array = np.asarray(organ_ls)
organ_array = np.delete(organ_array, np.where(organ_array == nr_cls[1]))

loc_array = np.asarray(loc_ls)
loc_array = np.delete(loc_array, np.where(loc_array == nr_cls[2]))

unique, counts = np.unique(ds_array, return_counts=True)
counts_dis[0,unique] = counts/ds_array.shape[0]

unique, counts = np.unique(organ_array, return_counts=True)
counts_organ[0,unique] = counts/organ_array.shape[0]

unique, counts = np.unique(loc_array, return_counts=True)
counts_loc[0,unique] = counts/loc_array.shape[0]
a = ds_array.shape[0]
b = organ_array.shape[0]
c = loc_array.shape[0]



#########validation###########
with open("D:/studium/MIML/radgraph/radgraph/premium_selected/detr_data_dev_selected_20.json", 'r') as f:
    ref_dict = json.load(f)

#126, 47, 80
ds_ls = []
organ_ls = []
loc_ls = []
for key,ls_ls in ref_dict.items():
    ls = np.asarray(ls_ls)
    ds_ls.extend(ls[:,0])
    organ_ls.extend(ls[:,1])
    loc_ls.extend(ls[:,2])

ds_array = np.asarray(ds_ls)
ds_array = np.delete(ds_array, np.where(ds_array == nr_cls[0]))

organ_array = np.asarray(organ_ls)
organ_array = np.delete(organ_array, np.where(organ_array == nr_cls[1]))

loc_array = np.asarray(loc_ls)
loc_array = np.delete(loc_array, np.where(loc_array == nr_cls[2]))

#print(ds_array.shape)

unique, counts = np.unique(ds_array, return_counts=True)
counts_dis[1,unique] = counts/ds_array.shape[0]

unique, counts = np.unique(organ_array, return_counts=True)
counts_organ[1,unique] = counts/organ_array.shape[0]

unique, counts = np.unique(loc_array, return_counts=True)
counts_loc[1,unique] = counts/loc_array.shape[0]

###calculate freq of occur for both case
a1 = ds_array.shape[0]
b1 = organ_array.shape[0]
c1 = loc_array.shape[0]

fre_occur_dis = (counts_dis[1,:] * a1 + counts_dis[0,:]*a)/(a1+a)
fre_occur_organ = (counts_organ[1,:] * b1 + counts_organ[0,:]*b)/(b1+b)
fre_occur_loc = (counts_loc[1,:] * c1 + counts_loc[0,:]*c)/(c1+c)

with open('fre_occur/fre_occur_dis_organ_loc_seletected_20.npy', 'wb') as f:
    np.save(f, fre_occur_dis)
    np.save(f, fre_occur_organ)
    np.save(f, fre_occur_loc)
#
# with open('fre_occur/fre_occur_dis_organ_loc.npy', 'rb') as f:
#     fre_occur_dis = np.load(f, fre_occur_dis)
#     fre_occur_organ = np.load(f, fre_occur_organ)
#     fre_occur_loc = np.load(f, fre_occur_loc)

#print(counts_loc[0,:])

####### draw #######
mode = 'disease'
width=0.2
if mode == 'disease':
    y0 = counts_dis[0,:]
    y1 = counts_dis[1,:]
    x0 = nr_cls[0]
elif mode == 'organ':
    y0 = counts_organ[0,:]
    y1 = counts_organ[1,:]
    x0 = nr_cls[1]
else:
    y0 = counts_loc[0,:]
    y1 = counts_loc[1,:]
    x0 = nr_cls[2]

plt.bar(np.arange(x0),y0,width,label = 'Train_'+mode)
plt.bar(np.arange(x0)+width,y1,width,label = 'Dev_'+mode)
plt.xlabel('Class of '+mode)
plt.ylabel('Frequency of occurrence')
plt.title('Occurrence frequency of '+mode+' in Train/Extend dataset')
plt.yticks(np.arange(0,0.02+max(np.max(y0),np.max(y1)),step=0.02))
plt.legend(loc='best')
plt.show()