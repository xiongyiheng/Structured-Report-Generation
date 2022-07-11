
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




mapping_observation = {
        "effusion":["effusions","effusion","infiltrate","infiltration"],
        "enlarged":["distended","elongation","elongated", "dilated","enlarged","large","larger","expanded","well - expanded","hyperexpansion","enlargement","increased","increase","exaggerated","widening","widened","distention","distension","increased enlarged"],
        "opacity":["hazinness","obscured","obscuration","opacity","opacities","opacified","opacifications","opacification","obscures","obscuring","indistinct","indistinctness","haziness","not less distinct","blurring"],
        "thickening":["thickening","thickenings","hyperlucent"],
        "drain":["drains","drain"],
        "normal":["normal","stable","unremarkable","top - normal","unchanged","better","improved","improvement"],
        "clear":["lucencies","clear","lucency"],
        "decrease":["decreased","lower","low","decrease"],
        "abnormal":["lesion","worsened","lesions","worse","abnormality","abnormalities","abnormal","deformity","disease","deformities","dysfunction"],
        "edema":["edema"],
        "malignancy":["malignancy","malignancies","cancer"],
        "hemorrhage":["hemorrhage","hemorrahge","bleeding","blood"],
        "congested":["congested","congestion","engorged"],
        "infection":["infectious","infection"],
        "sharp":["sharp","sharply"],
        "prominent":["prominence","prominent"],
        "consolidation": ["consolidation", "consolidations","consolidative"],
        "nodules": ["nodules", "nodule","nodular"],
        "pneumonic": ["pneumonic", "pneumonia"],
        "calcification": ["calcification","calcified","calcifications"],
        "tortuous": ["tortuous","tortuosity"],
        "atelectasis": ["atelectasis","atelectatic","atelectases"],
        "pneumothoraces": ["pneumothoraces","pneumothorace","pneumothorax","emphysema","aerated","pneumothoraces"],
        "fracture":["fractures","fracture","fractured"],
        "injury": ["injury", "injuries","trauma","traumas"],
    "position":["position","positions"],
    "visualized":["visualized"],
    "parenchymal":["parenchymal"],
"postoperative":["postoperative","post - operative"],
    "nondistended":["nondistended"],
    "confluent":["confluent"],
    "overlying":["overlying"],
    "catheter":["catheter","catheters"],
    "crowding":["crowding","crowding of"],
    "bilateral":["bilateral"],
    "demineralized":["demineralized"],
    "pic":["pic","picc"],
    "adenopathy":["adenopathy"],
    "kyphosis":["kyphosis"],
    "aicd":["aicd"],
"gallbladder":["gallbladder"],
"thoracolumbar":["thoracolumbar"],
"lymph node":["lymph node"],
    "nipple":["nipple"],
    "not well evaluated":["not well evaluated","not well assessed","not well seen","not seen","difficult to assess"],
    "operation":["operation"],
    "volume":["volume", "volumes"],
    "wire":["wire","wires"],
    "decompensation":["decompensation"],
"collapse":["collapse"],
    "defibrillator":["defibrillator"],
    "obstruction":["obstruction"],
    "granulomas":["granulomas"],
    "compression":["compression"],
    "object":["object"],
    "hardware":["hardware"],
    "port":["port","ports"],
    "tip":["tip","tips"],
    "atherosclerotic":["atherosclerotic"],
    "inflammatory":["inflammatory"],
    "loculated":["loculated"],
    "cardiomyopathy":["cardiomyopathy"],
    "fullness":["fullness"],
    "density":["density","densities"],
    "unfolded":["unfolded"],
    "vessel on end":["vessel on end"],
    "pacemaker":["pacemaker"],
    "ileus":["ileus"],
    "accentuation":["accentuation"],
    "pneumopericardium":["pneumopericardium"],
    "mucous":["mucous"],
    "scoliosis":["scoliosis"],
    "similar":["similar"],
    "blunting":["blunting"],
    "scarring":["scarring"],
    "sidehole":["sidehole","the side - port","side port", "side holes", "side ports"],
    "engorgement":["engorgement"],
    "copd":["copd"],
    "bubble":["bubble"],
    "to":["to"],
    "cephalization":["cephalization"],
    "pressure":["pressure"],
    "cardiogenic":["cardiogenic"],
    "silhouetting":["silhouetting","silhouette"],
    "replacement":["replacement","replacements"],
    "sheath":["sheath"],
    "sarcoidosis":["sarcoidosis"],
    "area":["area","areas"],
    "more":["more"],
    "multiple":["multiple","several"],
    "poor definition":["poor definition"],
    "markings":["markings"],
    "intact":["intact"],
    "resection":["resection"],
    "evidence of prior smoking":["evidence of prior smoking"],
    "flattened":["flattened","flattening"],
    "fibrotic":["fibrotic","fibrosis"],
    "device":["device","devices","hardware"],
    "free":["free"],
    "hypertension":["hypertension"],
    "moderate":["moderate","milder"],
    "aeration":["aeration"],
    "component":["component"],
      "process":["process","processes"],
      "tube":["tube","tubes"],
      "clip":["clip","clips"],
      "high":["high"],
      "air bronchograms":["air bronchograms"],
      "borderline":["borderline","border","borders"],
      "one - third":["one - third"],
      "overload":["overload"],
      "surgery":["surgery"],
      "vasculature":["vasculature"],
      "change":["change","changes"],
      "chf":["chf"],
      "small":["small"],
      "within":["within"],
      "projection":["projection"],
      "wedging":["wedging"],
      "redistribution":["redistribution"],
      "similar to prior":["similar to prior"],
      "pronounced":["pronounced"],
      "radiodensity":["radiodensity"],
      "greater":["greater"],
      "infiltrates":["infiltrates"],
      "displacement":["displacement","displacements","shift","deviation"],
      "reaction to inhaled substances":["reaction to inhaled substances"],
      "thoracocentesis":["thoracocentesis"],
      "extensive":["extensive"],
      "obliterating":["obliterating"],
      "distance":["distance"],
      "degenerative":["degenerative"],
      "failure":["failure"],
      "migrated":["migrated"],
      "fat":["fat"],
      "pnemonia":["pnemonia"],
      "appearance":["appearance"],
      "wedge resection":["wedge resection"],
      "underinflated":["underinflated"],
      "thyroidectomy":["thyroidectomy"],
      "bulging":["bulging","inflated"],
      "air":["air","gas"],
      "sequela":["sequela"],
      "lead":["lead","leads"],
      "size":["size"],
      "findings":["findings"],
      "structure":["structure","structures"],
      "elevate":["elevate","elevation","elevated"],
      "obliteration":["obliteration"],
      "suture":["suture","sutures"],
      "loculation":["loculation"],
      "dahboff":["dahboff"],
      "its":["its"],
      "resorption":["resorption"],
      "it":["it"],
      "examination":["examination"],
      "radiograph":["radiograph"],
      "lymphadenopathy":["lymphadenopathy"],
      "line":["line"],
      "origin":["origin"],
      "aspiration":["aspiration"],
      "asymmetry":["asymmetry"],
      "loss":["loss"],
      "ij":["ij"],
      "high - positioned":["high - positioned"],
      "separation":["separation"],
      "implement":["implement"],
      "rod":["rod"],
      "pacer":["pacer"],
      "pneumomediastinum":["pneumomediastinum"]
}


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
           "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar","carina"],
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
       "spine":["spine","vertebral","spinal","paraspinal","t 12 vertebral","thoracolumbar","thoracolumbar junction","carina"],
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
                                                    OBS_with_modify = obs_modified  #cancer
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
                                            OBS_with_modify = obs_modified  # cancer
                                        else:
                                            OBS_with_modify = entity_4["tokens"].lower() + " " + obs_modified  # 3cm cancer
                                        #OBS_with_modify = obs_modified  # 3cm cancer

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
                    # if OBS_with_modify == 'limits':
                    #     print(key)
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

rem_ls = ['clear',"normal",'abnormal']

for key_organ in final_dict.keys():
    for key, ls in final_dict[key_organ].items():
        #key:'left'         ls:['large','larged']
        #mapping
        for i in range(len(ls)):
            af_mapping = mapping_name(mapping_observation,ls[i])
            if af_mapping == None:
                af_mapping = ls[i]
            ls[i] = af_mapping
        ### remove all general OB in label

        ls = list(set(ls))
        for e in ls:
            if e in rem_ls: #if "clear" in ls
                ls.remove(e)
                #print(e)

        final_dict[key_organ][key] = ls
print(len(final_dict.keys()))

for key_organ in final_dict.keys():
    for key, ls in final_dict[key_organ].items():
        for e in ls:
            if e in rem_ls: #if "clear" in ls
                ls.remove(e)
                print(e)

        final_dict[key_organ][key] = ls

a_file = open("organ_loc_ob.json", "w")
json.dump(final_dict, a_file)
a_file.close()




#print(final_dict.keys())
#print(observation_dict["lung"])
