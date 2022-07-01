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

mapping = {"lung":["lung","pulmonary","lungs","midlung","subpulmonic","pulmonic"],
           "pleural":["pleural","plerual"],
            "heart":["heart","cardiac","retrocardiac"],
           "mediastinum":["mediastinal","cardiomediastinal","mediastinum"],
           "lobe":["lobe","lobar","lobes"],
            "hilar":["hilar","hila","hilus","suprahilar"],
           "vascular":["vascular","coronary",'internal jugular',"intravascular","vasculature","bronchovascular","venous","aortic","vein","arteries","vasculatiry","aorta","artery","vessel","vessels", "vascularity", "superior vena cava", "jugular"],
           "chest":["chest","thorax","pectus", "intrathoracic","hemithorax"],
           "cardiopulmonary":["cardiopulmonary"],
           "basilar":["bibasilar","basilar","base","bibasal","basal","bases"],
           "diaphragm":["hemidiaphragm","diaphragm","hemidiaphragms","diaphragms"],
           "rib":["rib","ribs", "ribcage", "costophrenic"],
           "stomach":["stomach","gastric"],
           "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar"],
           "carina":["carina"],
           "esophagus":["esophagus"],
           "apex":["apex","apical"],
           "osseous":["osseous"],
           "esophagogastric":["esophagogastric","gastroesophageal"],
           "bony":["bony","bones","skeletal"],
           "quadrant":["quadrant"],
           "sternal":["sternal","thoracic"],
           "svc":["svc"],
           "subclavian":["subclavian","clavicle"],
           "atrium":["atrium"],
           "valve":["valve", "valvular"],
           "pericardial":["pericardial","perihilar"],
           "skin":["skin","cutaneous"],
           "airway":["airway","airspace"],
           "institial":["institial"],
           "interstitial":["interstitial"],
           "parenchymal":["parenchymal"],
           "line":["line", "lines"],
           "fissure":["fissure"],
           "junction":["junction"],
           "lingular":["lingular","lingula"],
           "infrahilar":["infrahilar"],
           "biapical":["biapical"],
           "neck":["neck"],
           "apical":["apical"],
           "paratracheal":["paratracheal", "trachea","peribronchial","bronchial"],
           "thyroid":["thyroid"],
           "ge":["ge"],
           "axillary":["axillary","axilla"],
           "ventricle":["ventricle","ventricular","cavoatrial","biventricular","cavoatrial junction"],
           "left arm":["left arm"],
           "scapula":["scapula"],
           "subcutaneous":["subcutaneous", "subcutaneus"],
           "soft tissues":["soft tissues", "soft tissue"],
           "ij":["ij"],
           "sheath":["sheath"],
           "alveolar":["alveolar"],
           "pylorus":["pylorus"],
           "subsegmental":["subsegmental"],
           "lumbar":["lumbar"],
           "abdomen":["abdomen"],
           "duodenum":["duodenum"],
           "fundus":["fundus"],
           "inlet":["inlet"],
           "subdiaphragmatic":["subdiaphragmatic"],
           "cervical":["cervical"],
           "zone":["zone"],
           "volume":["volume", "volumes"],
           "tube":["tube","tubes"],
           "bowel":["bowel"],
           "annulus":["annulus"],
           "cavitary":["cavitary"],
           "interstitium":["interstitium"],
           "cage":["cage"]
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

mapping_subpart = {
        "lung":["lung","pulmonary","lungs","midlung","subpulmonic","pulmonic"],
       "pleural":["pleural","plerual"],
        "heart":["heart","cardiac","retrocardiac"],
       "mediastinum":["mediastinal","cardiomediastinal","mediastinum"],
       "lobe":["lobe","lobar","lobes"],
        "hilar":["hilar","hila","hilus","perihilar","suprahilar"],
       "vascular":["vascular","coronary",'internal jugular',"intravascular","vasculature","bronchovascular","venous","aortic","vein","arteries","vasculatiry","aorta","artery","vessel","vessels", "vascularity", "superior vena cava", "jugular"],
       "chest":["chest","thorax","pectus", "intrathoracic","hemithorax"],
       "cardiopulmonary":["cardiopulmonary"],
       "basilar":["bibasilar","basilar","base","bibasal","basal","bases"],
       "diaphragm":["hemidiaphragm","diaphragm","hemidiaphragms","diaphragms"],
       "rib":["rib","ribs", "ribcage", "costophrenic"],
       "stomach":["stomach","gastric"],
       "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar","thoracolumbar junction"],
       "carina":["carina"],
       "esophagus":["esophagus"],
       "apex":["apex","apical"],
       "osseous":["osseous"],
       "esophagogastric":["esophagogastric","gastroesophageal"],
       "bony":["bony","bones","skeletal"],
       "quadrant":["quadrant"],
       "sternal":["sternal","thoracic"],
       "svc":["svc"],
       "subclavian":["subclavian","clavicle"],
       "atrium":["atrium"],
       "valve":["valve", "valvular"],
       "pericardial":["pericardial"],
       "skin":["skin","cutaneous"],
       "airway":["airway","airspace"],
       "institial":["institial"],
       "interstitial":["interstitial"],
       "parenchymal":["parenchymal"],
       "line":["line", "lines"],
       "fissure":["fissure"],
       "junction":["junction", "ge junction"],
       "lingular":["lingular","lingula"],
       "infrahilar":["infrahilar"],
       "biapical":["biapical"],
       "neck":["neck"],
       "apical":["apical"],
       "paratracheal":["paratracheal", "trachea","peribronchial","bronchial"],
       "thyroid":["thyroid"],
       "ge":["ge"],
       "axillary":["axillary","axilla"],
       "ventricle":["ventricle","ventricular","cavoatrial","biventricular","cavoatrial junction"],
       "left arm":["left arm"],
       "scapula":["scapula"],
       "subcutaneous":["subcutaneous", "subcutaneus"],
       "soft tissues":["soft tissues", "soft tissue"],
       "ij":["ij"],
       "sheath":["sheath"],
       "alveolar":["alveolar"],
       "pylorus":["pylorus"],
       "subsegmental":["subsegmental"],
       "lumbar":["lumbar"],
       "abdomen":["abdomen"],
       "duodenum":["duodenum"],
       "fundus":["fundus"],
       "inlet":["inlet"],
       "subdiaphragmatic":["subdiaphragmatic"],
       "cervical":["cervical"],
       "zone":["zone", "area", "areas", "region"],
       "volume":["volume","volumes"],
       "tube":["tube","tubes"],
       "bowel":["bowel"],
       "annulus":["annulus"],
       "cavitary":["cavitary"],
       "interstitium":["interstitium"],
       "cage":["cage"],

        "right": ["right", "right - sided", "right sided",'the right - sided', "right side"],
        "left": ["left", "left - sided", "left-sided"],
        "contour": ["contour", "silhouette", "silhouettes", "contours"],
        "structure": ["structure", "structures"],
        "surface": ["surface", "surfaces"],
        "bilateral": ["bilateral", "bilaterally"],
        "lower":["lower","low", "descending", "below","beneath"],
        "mid":["mid","central","median","middle","midline"],
        "upper":["upper","superiorly", "above","3.6 cm above","2.9 cm above","1.8 cm above","5 cm above","6 cm above","2.8 cm above","4.6 cm above","4.7 cm above"],
        "major":["major"],
        "small":["small"],
        "great":["great"],
    "content":["content", "contents"],
    "angle":["angle"],
    "related":["related"],
    'azygos':['azygos'],
    "medially":["medially","internal","interspace", "within","medial","in"],
    "tip":["tip"],
    "cp":["cp"],
    "anterior":["anterior"],
    "body":["body","bodies"],
    "unchanged":["unchanged"],
    "configuration":["configuration"],
    "fourth":["fourth"],
    "pedicle":["pedicle"],
    "arch":["arch"],
    "mitral":["mitral"],
    "grossly":["grossly"],
    "adjacent":["adjacent"],
    "lateral":["lateral",'laterally'],
    "excavatum":["excavatum"],
    "compartment":["compartment"],
    "edema":["edema"],
    "remainder":["remainder"],
    "third":["third"],
    "nondistended":["nondistended"],
    "wall":["wall"],
    "border":["border","margin","periphery"],
    "nipple":["nipple"],
    "sinus":["sinus","sinuses"],
    "minor":["minor"],
    "size":["size"],
    "part":["part", "parts", "portion", "position","field"],
    "proximally":["proximally","proximal"],
    "diameter":["diameter"],
    "distal":["distal"],
    "first":["first"],
    "ac":["ac"],
    'multiple':['multiple'],
    "posterior":["posterior"],
"brachiocephalic":["brachiocephalic"],
    "mid - to - distal":["mid - to - distal"],
"cardia":["cardia"],
    "tricuspid":["tricuspid"],
    "of t 5 through t 9":["of t 5 through t 9"]
}


with open('D:/studium/MIML/radgraph/radgraph/train.json', 'r') as f:
    data = json.load(f)

organs = []
observation_dict = create_dict_from_dict(mapping)

#print(observation_dict)


#{"lung":None,
#   }



#print observation

###   extract the labels with "ANAT-DP"   ###
# for key in data.keys():
#     new_dict = data[key]
#     entities = new_dict['entities']
#     for new_key in entities.keys():
#         entity = entities[new_key]
#         if entity['label'] == 'ANAT-DP':
#             relations = entity['relations']
#             is_contain_modify = False
#             for i in range(len(relations)):
#                 if relations[i][0] == 'modify':
#                     is_contain_modify = True
#             if is_contain_modify == False:
#                 if entity['tokens'][-1] == 's' and entity['tokens'][-3:-1] != "ou" :
#                     entity['tokens'] = entity['tokens'][:-1]
#                 entity['tokens'] = mapping_name(mapping,entity['tokens'])
#                 if entity['tokens'] != None:
#                     organs.append(entity['tokens'].lower())

# ### statistic and post-process###
# result = Counter(organs)
# sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
# print(sorted_result)
#
#
# #output distince outputs
# organs = list(set(organs))

##########################################
########## extract the organs ######
##########################################

# for key in data.keys():
#     new_dict = data[key]
#     entities = new_dict['entities']
#     for new_key in entities.keys():
#         entity = entities[new_key]
#         relations = entity['relations']
#         for i in range(len(relations)):
#             if relations[i][0] == "located_at":
#                 is_contain_modify = False
#                 relations_2 = entities[relations[i][1]]['relations']
#                 for j in range(len(relations_2)):
#                     if relations_2[j][0] == 'modify':
#                         is_contain_modify = True
#                 if is_contain_modify == False:
#                 #if entity['tokens'] == 'chest':
#                     #print(new_dict['text'])
#                     if entities[relations[i][1]]['tokens'] == None:
#                         print(entities[relations[i][1]])
#                     lower_token = entities[relations[i][1]]['tokens'].lower()
#                     after_mapping = mapping_name(mapping,lower_token)
#                     if after_mapping != None:
#                         organs.append(after_mapping.lower())
#
#                     #if lower_token[-1] == 's':
#                         #organs.append(lower_token[:-1])
#                     # else:
#                     #     organs.append(lower_token)
#
# ### statistic and post-process###
# result = Counter(organs)
# sorted_result = sorted(result.items(), key=lambda x:x[1], reverse=True)
# print(sorted_result)
#
#
# #output distince outputs
# organs = list(set(organs))








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



