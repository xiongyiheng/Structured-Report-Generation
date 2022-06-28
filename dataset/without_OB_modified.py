
import json
import os
from collections import Counter

####### output dictionary ######
#in format: {"hilar":{"hilar":{},"contours":{3cm cancer, opacaties}}}

### global settings ###

OB_modified = False



def del_space(s):
    #input: string
    #output: children-string after the space

    if " " in s:
        output = s.split()[-1]

        return output
    else:
        return s

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




mapping = {"lung":["lung","pulmonary","lungs","midlung","subpulmonic"],
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
           "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar"],
           "carina":["carina"],
           "esophagus":["esophagus"],
           "apex":["apex"],
           "osseous":["osseous"],
           "esophagogastric":["esophagogastric","gastroesophageal"],
           "bony":["bony","bones","skeletal"],
           "quadrant":["quadrant"],
           "sternal":["sternal","thoracic"],
           "svc":["svc"],
           "subclavian":["subclavian","clavicle"],
           "atrium":["atrium"],
           "valve":["valve"],
           "pericardial":["pericardial"],
           "skin":["skin","cutaneous"],
           "airway":["airway","airspace"],
           "institial":["institial"],
           "interstitial":["interstitial"],
           "parenchymal":["parenchymal"],
           "line":["line", "lines"],
           "fissure":["fissure"],
           "junction":["junction", "cavoatrial junction"],
           "lingular":["lingular","lingula"],
           "valvular":["valvular"],
           "infrahilar":["infrahilar"],
           "biapical":["biapical"],
           "neck":["neck"],
           "apical":["apical"],
           "paratracheal":["paratracheal", "trachea","peribronchial","bronchial"],
           "thyroid":["thyroid"],
           "ge":["ge"],
           "axillary":["axillary","axilla"],
           "ventricle":["ventricle","ventricular","cavoatrial","biventricular'"],
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
           "volumes":["volumes"],
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
        "enlarged":["enlarged","large","larger","expanded","well - expanded","hyperexpansion","enlargement",
                    "increased","increase","exaggerated","widening","widened","distention","distension","hyperinflated",
                    "hyperexpanded"],
        "opacity":["opacity","opacities","opacified","opacification","obscures","obscuring","indistinct","indistinctness","haziness","not less distinct","blurring"],
        "thickening":["thickening","thickenings"],
        "drain":["drains","drain","fluid"],
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
        "lung":["lung","pulmonary","lungs","midlung","subpulmonic"],
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
       "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar"],
       "carina":["carina"],
       "esophagus":["esophagus"],
       "apex":["apex"],
       "osseous":["osseous"],
       "esophagogastric":["esophagogastric","gastroesophageal"],
       "bony":["bony","bones","skeletal"],
       "quadrant":["quadrant"],
       "sternal":["sternal","thoracic"],
       "svc":["svc"],
       "subclavian":["subclavian","clavicle"],
       "atrium":["atrium"],
       "valve":["valve"],
       "pericardial":["pericardial"],
       "skin":["skin","cutaneous"],
       "airway":["airway","airspace"],
       "institial":["institial"],
       "interstitial":["interstitial"],
       "parenchymal":["parenchymal"],
       "line":["line", "lines"],
       "fissure":["fissure"],
       "junction":["junction", "cavoatrial junction"],
       "lingular":["lingular","lingula"],
       "valvular":["valvular"],
       "infrahilar":["infrahilar"],
       "biapical":["biapical"],
       "neck":["neck"],
       "apical":["apical"],
       "paratracheal":["paratracheal", "trachea","peribronchial","bronchial"],
       "thyroid":["thyroid"],
       "ge":["ge"],
       "axillary":["axillary","axilla"],
       "ventricle":["ventricle","ventricular","cavoatrial","biventricular'"],
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
       "volumes":["volumes"],
       "tube":["tube","tubes"],
       "bowel":["bowel"],
       "annulus":["annulus"],
       "cavitary":["cavitary"],
       "interstitium":["interstitium"],
       "cage":["cage"],
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




final_dict={}



with open('D:/studium/MIML/radgraph/radgraph/train_add_sug.json', 'r') as f:
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
                                                if OB_modified == False: #no ob_modified
                                                    OBS_with_modify = entity_4["tokens"].lower()  #3cm cancer
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
                                        OBS_with_modify = obs_modified  # 3cm cancer

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

                        organ_modify_copy = organ_modify
                        organ_modify = mapping_name(mapping_subpart, organ_modify_copy)
                        if organ_modify == None:
                            organ_modify = organ_modify_copy
                        # if OBS_with_modify:
                        #     OBS_with_modify = del_space(OBS_with_modify)


                        #organ_modify = organ_lower_token
                                #### mapping organs' name ####
                        #{organ_after_mapping:[]}

                    output_dict = {organ: {organ_modify.lower(): [OBS_with_modify]}}
                    final_dict = update_dict(final_dict,output_dict)

# for key, ls in final_dict['alveolar'].items():
#     #key:'left'         ls:['large','larged']
#     #mapping
#     for i in range(len(ls)):
#         af_mapping = mapping_name(mapping_observation,ls[i])
#         if af_mapping == None:
#             af_mapping = ls[i]
#         ls[i] = af_mapping
#     #distinct
#     result = Counter(ls)
#     sorted_result = sorted(result.items(), key=lambda x: x[1], reverse=True)
#
#     print({key:sorted_result})
    #print('/n')


for key_organ in final_dict.keys():
    for key, ls in final_dict[key_organ].items():
        #key:'left'         ls:['large','larged']
        #mapping
        for i in range(len(ls)):
            af_mapping = mapping_name(mapping_observation,ls[i])
            if af_mapping == None:
                af_mapping = ls[i]
            ls[i] = af_mapping
        final_dict[key_organ][key] = list(set(ls))
print(len(final_dict.keys()))

a_file = open("organ_loc_ob.json", "w")
json.dump(final_dict, a_file)
a_file.close()




#print(final_dict.keys())
#print(observation_dict["lung"])

