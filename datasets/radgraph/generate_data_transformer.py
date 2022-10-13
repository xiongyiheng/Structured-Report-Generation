"""
generate dataset(label) for transformer training
output format in json file like:
"report id":
                {lung:  {
                        lung:{ 'edema', 'no clear',...
                               }
                        left:{'no edema', 'clear'
                        }
                heart:{...
                           }
                labels:[1,0,1], [2,3,4]....

                }
"""

import json
import os
from collections import Counter


global count
global max_nr
global N
max_nr=0
count=0
OB_modified = False
splite = "dev"
N = 30




def del_space(s):
    """
    input: string
    output: children-string after the space
    """
    if " " in s:
        output = s.split()[-1]

        return output
    else:
        return s

def write_json(data,file_path):#'D:/studium/MIML/radgraph/radgraph/train_add_sug.json'
    with open(file_path, 'w') as outfile:
        json.dump(data, outfile)

def gen_token_for_each_obs(organ,loc,OBS_with_modify,obs_ls,organ_ls,loc_ls,cls_ls):
    """
    if normal -> other attributes in this sub part are all 0 and masked = 1 (not sure whether to use or not)
    """
    if OBS_with_modify in obs_ls:
        ind_OBS = obs_ls.index(OBS_with_modify)
    else:
        return None
    if organ in organ_ls:
        ind_organ = organ_ls.index(organ)
    else:
        return None
    if loc in loc_ls:
        ind_loc = loc_ls.index(loc)
    else:
        return None
    if [ind_OBS,ind_organ,ind_loc] in cls_ls:
        return [ind_OBS,ind_organ,ind_loc]
    else:
        return None


def gen_dataset(dataset,id,token_ls_each_report,dis_ls,organ_ls,loc_ls):
    """
    generate dataset (label) like:
    {"report id":
    [
        [1,2,3]
        [1,5,9]
        ...
        [0,0,0]

    ]
        
    {"report id2":][...]}
    """
    global max_nr
    global N

    none_obs = [len(dis_ls),len(organ_ls),len(loc_ls)]  # placeholder
    #print(none_obs)

    # count the max_nr of observations in a img
    if len(token_ls_each_report)> N:#skip the reports containing > 30 diseases
        print('over')
        #print(len(token_ls_each_report))
        return dataset

    if len(token_ls_each_report)>max_nr:
        max_nr = len(token_ls_each_report)

    # add placeholders for none objects
    nr_none = N - len(token_ls_each_report)

    for i in range(nr_none): #nr_none=10
        token_ls_each_report.append(none_obs)

    # drop a issue when len(tokens) !=N
    assert len(token_ls_each_report)==N

    ### rename the keys
    a = id.split('/')
    id = a[-3] + '/' + a[-2] + '/' + a[-1]


    dataset[id] = token_ls_each_report

    return dataset


def update_dict(final_dict, dic):
    """
    update final_dict values with values in dic

    dic = {"organ":{"organ_modify":[3cm cancer]}}
    output = {organ:{organ_modify:[clear, 3cm cancer]}}
    """
    for key, ls in dic.items():
        if key not in final_dict.keys():
            final_dict.update(dic)
        else:
            inner_dic = dic[key]
            inner_final_dic = final_dict[key]
            for inner_key, inner_ls in inner_dic.items():  # "organ_modify":[clear]
                if inner_key not in inner_final_dic.keys():
                    # inner_final_dic.update(inner_dic)   #need check if final_dict also updates
                    final_dict[key][inner_key] = inner_dic[inner_key]  # add the first [clear]
                else:
                    final_dict[key][inner_key].append(dic[key][inner_key][0])

    return final_dict


# p = os.getcwd().
def mapping_name(dict,value):
    """
    mapping each OB into the mapped OB according to dict
    :param mapping_observation: an OB
    :param dic: the mapping dict
    :return: the mapped OB
    """
    for key,ls in dict.items():
        if value in ls:
            return key
    # if cant find the key
    return None

# def filter_out_observations(input_dict):
#     """
#     input: dict= {"lung":[[],[]],
#               "...":...}
    
#     output: dict={"lung":["clear","normal"],
#                    "...": []}
#     """
#     out_dict = dict.fromkeys(input_dict, [])
#     for key,ls in input_dict.items():
#         obser_ls = []  # value for output like ["clear","normal"]
#         for l in ls:
#             obser_ls.append(l[0])
#         # distinct output
#         #obser_ls = list(set(obser_ls))
#         out_dict[key] = obser_ls

#     return out_dict

def create_dict_from_dict(dic):
    """
    Create dict from a dict with same values
    """
    outdic = dic.copy()
    for key,ls in dic.items():
        outdic[key] = []
    # if cant find the key
    return outdic

# def mapping_observations(mapping_observation,dic):
#     #output:
#     #   dict{"lung":[normal,effusion],
#     #              "...":[...]}
#     out_dict = create_dict_from_dict(dic)
#
#     for key, ls in dic.items():  #ls = ["effusions","effusion"]
#         for l in ls:
#             after_mapping = mapping_name(mapping_observation, l)
#             if after_mapping != None: #this oberservation exists in mapping dict
#                 out_dict[key].append(after_mapping)
#
#     return out_dict

def gen_init_dict_for_each_report(ref_dict,mask = False):
    """
    generate the initial version of dict for each report
    """

    output = ref_dict.copy()
    for key,ls in output.items():
        inner_dic = output[key]
        for key1, ls1 in inner_dic.items():
            for i in range(len(ls1)):
                ls1[i]=0
                # if mask:
                #     ls1[i]=0
                # else:
                #     if ls1[i] == "clear" or ls1[i] == "normal":
                #         ls1[i] = 0
                #     else:
                #         ls1[i]=0
    # if cant find the key
    return output



# a_file = open("organ_loc_ob.json", "r")
# ref_dict = a_file.read()
# a_file.close()
#
# a_file = open("organ_loc_ob.json", "r")
# ref_dict1 = a_file.read()
# a_file.close()
organ_loc_ob_json = "organ_loc_ob.json"

# with open(organ_loc_ob_json, 'r') as f:
#     ref_dict = json.load(f)
# with open(organ_loc_ob_json, 'r') as f:
#     ref_dict1 = json.load(f)

dataset = {}

final_dict={}
if splite == "train":
    splite_open = "train_add_sug"
if splite == 'dev':
    splite_open = 'dev'
if splite == 'extend':
    splite_open = 'extend+train_add_sug'



with open('D:/studium/mlmi-structured-report-generation/datasets/radgraph/mapping_delete_meaningless.json', 'r') as f:
    mapping_dict = json.load(f)
mapping_subpart = mapping_dict["mapping_subpart"]
mapping_observation = mapping_dict["mapping_observation"]
mapping = mapping_dict["mapping_organs"]

with open('D:/studium/MIML/radgraph/radgraph/'+splite_open+'.json', 'r') as f:
    data = json.load(f)

#organ_loc_ob_json = "organ_loc_ob_extend_20"#"organ_loc_ob.json"
organs = []
observation_dict = create_dict_from_dict(mapping)

#print(observation_dict)

def gen_lists():
    with open(organ_loc_ob_json, 'r') as f:
        ref_dict = json.load(f)

    loc_ls = []
    organ_ls = []
    dis_ls = []

    for key,in_dict in ref_dict.items():
        organ_ls.append(key)
        for loc,ls in in_dict.items():
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

    cls_ls = []
    for key, in_dict in ref_dict.items():
        idx_organ = organ_ls.index(key)
        for loc, ls in in_dict.items():
            idx_loc = loc_ls.index(loc)
            for l in ls:
                idx_dis = dis_ls.index(l)
                cls_ls.append([idx_dis, idx_organ, idx_loc])

    return dis_ls,organ_ls,loc_ls,cls_ls




#### generate lists for all diseases, organs and locations ####

# num_words_mapping = {}
# num_words_mapping['cls_ls'] = cls_ls
# num_words_mapping['dis_ls'] = dis_ls
# num_words_mapping['organ_ls'] = organ_ls
# num_words_mapping['loc_ls'] = loc_ls
with open("num_words_mapping_original.json", "r") as f:
    num_words_mapping = json.load(f)

dis_ls,organ_ls,loc_ls,cls_ls = num_words_mapping['dis_ls'],num_words_mapping['organ_ls'],num_words_mapping['loc_ls'],num_words_mapping['cls_ls']
print(len(cls_ls))

token_ls_each_report = []

##########################################
########## extract the observations ######
##########################################
for key in data.keys():  # key : "p18/p18004941/s58821758.txt"
    #print(key)
    new_dict = data[key]
    entities = new_dict['entities']
    # with open(organ_loc_ob_json, 'r') as f:
    #     ref_dict1 = json.load(f)
    # dict_each_report = gen_init_dict_for_each_report(ref_dict1)
    # with open(organ_loc_ob_json, 'r') as f:
    #     ref_dict1 = json.load(f)
    #masked_each_report = gen_init_dict_for_each_report(ref_dict1,mask=True)

    for new_key in entities.keys():
        entity = entities[new_key]     #entity can be "6":{}      "cancer"
        relations = entity['relations']
        for i in range(len(relations)):
            if relations[i][0] == "located_at":
                is_contain_modify = False
                relations_2 = entities[relations[i][1]]['relations']  #entities[relations[i][1]]  "lung"
                for j in range(len(relations_2)):
                    if relations_2[j][0] == 'modify':
                        is_contain_modify = True
                if is_contain_modify == False:
                    #  organs tokens, Upper case -> lower case
                    organ_lower_token = entities[relations[i][1]]['tokens'].lower()      #lung


                    ### modify the organ_lower_token
                    found_modify_ANAT = False
                    for new_key_3 in entities.keys():
                        entity_3 = entities[new_key_3]  ### "left"
                        relations_3 = entity_3['relations']
                        for k in range(len(relations_3)):

                            if relations_3[k][0] == "modify" and relations_3[k][1] == relations[i][1]:
                                found_modify_ANAT = True

                                #### handle with "modify" OBS
                                found_modify_OBS = False
                                for new_key_4 in entities.keys():
                                    entity_4 = entities[new_key_4]  # entity can be "6":{}    "3cm"
                                    label_4 = entity_4['label']

                                    if label_4[:3] == "OBS":
                                        relations_4 = entity_4['relations']
                                        for l in range(len(relations_4)):
                                            if relations_4[l][0] == "modify" and relations_4[l][1] == new_key: #found OBS_modify and found Organ_modify
                                                found_modify_OBS = True
                                                obs_modified = mapping_name(mapping_observation,entity["tokens"].lower()) #entity['tokens'] = 'cancer'
                                                if obs_modified == None:# if it couldnot find its name in values of OBS_mapping
                                                    obs_modified = entity["tokens"].lower()
                                                if OB_modified == False: #no ob_modified
                                                    OBS_with_modify = obs_modified  #3cm cancer
                                                else:
                                                    OBS_with_modify = entity_4["tokens"].lower() + " " + obs_modified  # 3cm cancer

                                                organ_after_mapping = mapping_name(mapping,organ_lower_token)
                                                if organ_after_mapping == None:
                                                    organ_after_mapping = organ_lower_token #lung
                                                organ_modify = entity_3['tokens'] # left
                                                organ=organ_after_mapping # lung



                                if not found_modify_OBS: # not found OBS_modify but found Organ_modify
                                    obs_modified = mapping_name(mapping_observation,entity["tokens"].lower())
                                    if obs_modified == None:  # if it couldnot find its name in values of OBS_mapping
                                        obs_modified = entity["tokens"].lower()      # cancer
                                    OBS_with_modify = obs_modified

                                    organ_after_mapping = mapping_name(mapping, organ_lower_token)
                                    if organ_after_mapping == None:
                                        organ_after_mapping = organ_lower_token  # lung
                                    organ_modify = entity_3['tokens']  # left
                                    organ = organ_after_mapping



                    if not found_modify_ANAT:  # if not found modify_organ

                        #### handle with "modify" OBS
                        found_modify_OBS = False
                        for new_key_4 in entities.keys():
                            entity_4 = entities[new_key_4]  # entity can be "6":{}    "3cm"
                            label_4 = entity_4['label']

                            if label_4[:3] == "OBS":
                                relations_4 = entity_4['relations']
                                for l in range(len(relations_4)):
                                    if relations_4[l][0] == "modify" and relations_4[l][1] == new_key:  # found OBS_modify and found Organ_modify
                                        found_modify_OBS = True
                                        obs_modified = mapping_name(mapping_observation, entity["tokens"].lower())  # entity['tokens'] = 'cancer'
                                        if obs_modified == None:  # if it couldnot find its name in values of OBS_mapping
                                            obs_modified = entity["tokens"].lower()
                                        if OB_modified == False:  # no ob_modified
                                            OBS_with_modify = obs_modified  #cancer
                                        else:
                                            OBS_with_modify = entity_4["tokens"].lower() + " " + obs_modified  # 3cm cancer

                                        organ_after_mapping = mapping_name(mapping, organ_lower_token)
                                        if organ_after_mapping == None:
                                            organ_after_mapping = organ_lower_token  # lung
                                        organ_modify = organ_after_mapping  # lung
                                        organ = organ_after_mapping  # lung

                        if not found_modify_OBS:  # not found OBS_modify nor found Organ_modify

                            obs_modified = mapping_name(mapping_observation, entity["tokens"].lower())
                            if obs_modified == None:  # if it couldnot find its name in values of OBS_mapping
                                obs_modified = entity["tokens"].lower()  # cancer
                            OBS_with_modify = obs_modified

                            organ_after_mapping = mapping_name(mapping, organ_lower_token)
                            if organ_after_mapping == None:
                                organ_after_mapping = organ_lower_token  # lung
                            organ_modify = organ_after_mapping  # lung
                            organ = organ_after_mapping  # lung


                        #organ_modify = organ_lower_token
                                #### mapping organs' name ####
                        #{organ_after_mapping:[]}
                    organ_modify_copy = organ_modify
                    organ_modify = mapping_name(mapping_subpart,organ_modify_copy)
                    # if organ_modify == None:
                    #     organ_modify = organ_modify_copy
                    # if OBS_with_modify:
                    #     OBS_with_modify = del_space(OBS_with_modify)
                    ### add OBS_label as output, in order to make sure that a OBS present or not in a report
                    OBS_label = entity['label'][-2:]
                    OBS_with_modify = mapping_name(mapping_observation, OBS_with_modify)
                    organ = mapping_name(mapping, organ)

                    #output_dict = {organ: {organ_modify.lower(): [OBS_with_modify]}}
                    if (OBS_label != 'DA') and organ_modify and OBS_with_modify and organ:
                        ### for each observation
                        token_each_obs = gen_token_for_each_obs(organ,organ_modify,OBS_with_modify,dis_ls,organ_ls,loc_ls,cls_ls)
                        if token_each_obs:
                            token_ls_each_report.append(token_each_obs)
    ### for each report ###
    dataset = gen_dataset(dataset,key,token_ls_each_report,dis_ls,organ_ls,loc_ls)
    token_ls_each_report = []

write_json(dataset, 'D:/studium/MIML/radgraph/radgraph/premium_selected/detr_data_'+splite+'.json')
print(max_nr)
                    #final_dict = update_dict(final_dict,output_dict)


# for key, ls in final_dict['mediastinum'].items():
#     print({key:ls})
    #print('/n')

#print(final_dict['lung'])
#print(observation_dict["lung"])

### pose-process observation_dict ###

# out_dict = filter_out_observations(observation_dict) #del DP/DA/U
# out_dict = mapping_observations(mapping_observation,out_dict)  #mapping variant observations' name into few observations
# #print(out_dict)
#
#
#
# #### print out the oberservation and its occurancy number according to organs ###
# for key, _ in out_dict.items():
#     result = Counter(out_dict[key])
#     sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
#     print({key:sorted_result})
    #print(sorted_result)

#print(observation_dict)# == observation_dict["lung"])
#print(out_dict["lung"])



