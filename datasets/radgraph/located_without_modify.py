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
        "enlarged":["enlarged","large","larger","expanded","well - expanded","hyperexpansion","enlargement","increased","increase","exaggerated","widening","widened","distention","distension"],
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


with open('D:/studium/MIML/radgraph/radgraph/train.json', 'r') as f:
    data = json.load(f)

organs = []
observation_dict = create_dict_from_dict(mapping)
##########################################
########## extract the observations ######
##########################################
for key in data.keys():  # key : "p18/p18004941/s58821758.txt"
    new_dict = data[key]
    entities = new_dict['entities']
    for new_key in entities.keys():
        entity = entities[new_key]     #entity can be "6":{}
        relations = entity['relations']
        for i in range(len(relations)):
            if relations[i][0] == "located_at":
                is_contain_modify = False
                relations_2 = entities[relations[i][1]]['relations']
                for j in range(len(relations_2)):
                    if relations_2[j][0] == 'modify':
                        is_contain_modify = True
                if is_contain_modify == False:
                    #  organs tokens, Upper case -> lower case
                    organ_lower_token = entities[relations[i][1]]['tokens'].lower()

                    #  observations tokens(lower case) and labels(present/ absent/ uncertain) in list
                    observ_output = [entity['tokens'].lower(),entity['label']]
                    for k, ls in mapping.items():
                        if organ_lower_token in ls:
                            #observation_dict[k] = observation_dict[k].insert(-1,observ_output)

                            # temp = []
                            # for l in observation_dict[k]:
                            #     temp.append(l)
                            # temp.append(observ_output)
                            # observation_dict[k] = temp
                            observation_dict[k].append(observ_output)

                    #if lower_token[-1] == 's':
                        #organs.append(lower_token[:-1])
                    # else:
                    #     organs.append(lower_token)
#print(observation_dict["lung"])

### pose-process observation_dict ###

out_dict = filter_out_observations(observation_dict) #del DP/DA/U
out_dict = mapping_observations(mapping_observation,out_dict)  #mapping variant observations' name into few observations
#print(out_dict)



#### print out the oberservation and its occurancy number according to organs ###
for key, _ in out_dict.items():
    result = Counter(out_dict[key])
    sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
    print({key:sorted_result})
    #print(sorted_result)

#print(observation_dict)# == observation_dict["lung"])
#print(out_dict["lung"])



