################################
########## generate dataset for DL training
########## output format in json file like:
##########  "report id":
#                   {lung:  {
#                           lung:{ 'edema', 'no clear',...
#                                   }
#                           left:{'no edema', 'clear'
#                            }
#                     heart:{...
#                           }
#                     labels:[1,0,1,0,......]
#
#                    }

import json
import os
from collections import Counter


def del_space(s):
    #input: string
    #output: children-string after the space

    if " " in s:
        output = s.split()[-1]

    return output


# p = os.getcwd().
def mapping_name(dict,value):
    for key,ls in dict.items():
        if value in ls:
            return key
    # if cant find the key
    return None

def filter_out_observations(input_dict):
    # input: dict= {"lung":[[],[]],
    #          "...":...}
    #
    # output: dict={"lung":["clear","normal"],
    #               "...": []}
    out_dict = dict.fromkeys(input_dict, [])
    for key,ls in input_dict.items():
        obser_ls = []  # value for output like ["clear","normal"]
        for l in ls:
            obser_ls.append(l[0])
        # distinct output
        #obser_ls = list(set(obser_ls))
        out_dict[key] = obser_ls

    return out_dict

def create_dict_from_dict(dic):
    outdic = dic.copy()
    for key,ls in dic.items():
        outdic[key] = []
    # if cant find the key
    return outdic

def mapping_observations(mapping_observation,dic):
    #output:
    #   dict{"lung":[normal,effusion],
    #              "...":[...]}
    out_dict = create_dict_from_dict(dic)

    for key, ls in dic.items():  #ls = ["effusions","effusion"]
        for l in ls:
            after_mapping = mapping_name(mapping_observation, l)
            if after_mapping != None: #this oberservation exists in mapping dict
                out_dict[key].append(after_mapping)

    return out_dict

def update_dict(final_dict,dic):
    #final_dict = {}
    #dic = {"organ":{"organ_modify":[3cm cancer]}}
    #output = {organ:{organ_modify:[clear, 3cm cancer]}}
    for key, ls in dic.items():
        if key not in final_dict.keys():
            final_dict.update(dic)
        else:
            inner_dic = dic[key]
            inner_final_dic = final_dict[key]
            for inner_key, inner_ls in inner_dic.items():   #"organ_modify":[clear]
                if inner_key not in inner_final_dic.keys():
                    #inner_final_dic.update(inner_dic)   #need check if final_dict also updates
                    final_dict[key][inner_key] = inner_dic[inner_key]    # add the first [clear]
                else:
                    final_dict[key][inner_key].append(dic[key][inner_key][0])

    return final_dict




mapping = {"lung":["lung","pulmonary","lungs"],
           "pleural":["pleural","plerual"],
            "heart":["heart","cardiac","retrocardiac"],
           "mediastinum":["mediastinal","cardiomediastinal","mediastinum"],
           "lobe":["lobe","lobar","lobes"],
            "hilar":["hilar","hila","hilus","perihilar"],
           "vascular":["vascular","vasculature","bronchovascular","venous","aortic","vein","arteries","vasculatiry","aorta","artery","vessel","vessels"],
           "chest":["chest","thorax","pectus"],
           "cardiopulmonary":["cardiopulmonary"],
            "basilar":["bibasilar","basilar","base","bibasal","basal","bases"],
           "diaphragm":["hemidiaphragm","diaphragm","hemidiaphragms"],
           "rib":["rib","ribs"],
           "stomach":["stomach","gastric"],
           "spine":["spine"]
            }







############ OBSERVATION ##########
mapping_observation = {
        "effusion":["effusions","effusion"],
        "enlarged":["enlarged","large","larger","expanded","well - expanded","hyperexpansion","enlargement","increased","increase","exaggerated","widening","widened","distention","distension","increased enlarged"],
        "opacity":["opacity","opacities","opacified","opacification","obscures","obscuring","indistinct","indistinctness","haziness","not less distinct","blurring"],
        "thickening":["thickening","thickenings"],
        "drain":["drains","drain"],
        "normal":["normal","stable","unremarkable","top - normal","unchanged"],
        "clear":["clear"],
        "decrease":["decreased","lower","low","decrease"],
        "abnormal":["abnormality","abnormalities","abnormal","deformity","disease","deformities"],
        "edema":["edema"],
        "malignancy":["malignancy","malignancies","cancer"],
        "hemorrhage":["hemorrhage","hemorrahge","bleeding"],
        "congested":["congested","congestion","engorged"],
        "infection":["infectious","infection"],
        "sharpe":["shape","sharply"],
        "prominent":["prominence","prominent"],
        "consolidation": ["consolidation", "consolidations"],
        "nodules": ["nodules", "nodule"],
        "pneumonic": ["pneumonic", "pneumonia","nodular"],
        "calcifications": ["calcifications","calcified"],
        "tortuous": ["tortuous","tortuosity"],
        "atelectasis": ["atelectasis","atelectatic","atelectases"],
        "pneumothoraces": ["pneumothoraces","pneumothorace"],
        "fracture":["fractures","fracture","fractured"],
        "injury": ["injury", "injuries","trauma","traumas"]

}

mappig_subpart = {
	"lung":["lung","pulmonary","lungs"],
	"pleural":["pleural","plerual"],
	"heart":["heart","cardiac","retrocardiac"],
	"mediastinum":["mediastinal","cardiomediastinal","mediastinum"],
	"vascular":["vascular","vasculature","bronchovascular","venous","aortic","vein","arteries","vasculatiry","aorta","artery","vessel","vessels"],
	"right": ["right", "right - sided", "right sided"],
	"left": ["left", "left - sided"],
	"contour": ["contour", "silhouette", "silhouettes", "contours"],
	"structure": ["structure", "structures"],
	"surface": ["surface", "surfaces"],
	"bilateral": ["bilateral", "bilaterally"],
	"base":["base", "bases"],
	"lower":["lower"],
	"mid":["mid"],
	"upper":["upper"],
	"volume": ["volume", "volumes"]
}


ref_dict = {'lung':{'lung':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'],
            'volume':['decrease'],
         'left':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'], 'right':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'], 'lower':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'], 'mid':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'],  'upper':['edema', 'clear', 'consolidation', 'enlarged', 'normal', 'abnormal', 'opacity', 'effusion', 'nodule', 'pneumonic', 'atelectasis'], 'vascular':['normal', 'abnormal', 'congested'], 'base':['atelectasis', 'opacity']},
        'pleural':{'pleural':['normal', 'abnormal', 'effusion', 'thickening', 'drain'], 'right':['normal', 'abnormal', 'effusion', 'thickening', 'drain'], 'left':['normal', 'abnormal', 'effusion', 'thickening', 'drain'], 'bilateral':['normal', 'abnormal', 'effusion', 'thickening', 'drain'], 'surface':['normal', 'abnormal', 'clear', 'effusion']},
        'heart':{'heart':['normal', 'abnormal', 'opacity', 'atelectasis', 'consolidation'],
                 'size':['normal','abnormal','enlarged'], 'contour':['normal', 'abnormal', 'enlarged'],
                'left':['normal', 'abnormal', 'opacity', 'atelectasis', 'consolidation']},
        'mediastinum':{'mediastinum':['normal', 'abnormal', 'enlarged', 'shift'],
                       'contour':['normal', 'abnormal', 'enlarged'],
                       'structure':['normal', 'abnormal']},
        'vascular':{'vascular':['congested', 'calcification', 'crowding']}
}

dataset = {}

final_dict={}



with open('~/MIML/radgraph/train.json', 'r') as f:
    data = json.load(f)

organs = []
observation_dict = create_dict_from_dict(mapping)

#print(observation_dict)

##########################################
########## extract the observations ######
##########################################
for key in data.keys():  # key : "p18/p18004941/s58821758.txt"
    new_dict = data[key]
    entities = new_dict['entities']
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
                                                OBS_with_modify = entity_4["tokens"].lower() + " " +obs_modified  #3cm cancer



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
                                        OBS_with_modify = entity_4["tokens"].lower() + "<>" + obs_modified  # 3cm cancer

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

                    output_dict = {organ: {organ_modify.lower(): [OBS_with_modify]}}
                    final_dict = update_dict(final_dict,output_dict)

    dataset = {key:
                   {}

               }



for key, ls in final_dict['mediastinum'].items():
    print({key:ls})
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




