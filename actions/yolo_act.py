# yolo_act.py
from ultralytics import YOLO
from utils.location import find_bbox_centroid
import itertools
from itertools import permutations
import math

def standalone_yolo_act(image, confidence):
    model = YOLO('yolo_act.pt')
    detection = model.predict(image, conf=confidence)
    action_out = [{'class': box.cls.item(), 'action': detection[0].names[box.cls.item()],
               'xyxy': box.xyxy.tolist()[0], 'conf': box.conf.item()} for box in detection[0].boxes]
    i = 0
    while i < len(action_out):
        for object in action_out:
            object.update({'id': str(i)})
            i+=1
            
    for object in action_out:
        centroid = find_bbox_centroid(object['xyxy'][0], object['xyxy'][1], object['xyxy'][2], object['xyxy'][3])
        object.update({'centroid': centroid})

    return action_out

#-------

def assign(raw_results,action_out):
    person_list = []
    action_list = []
    pairs = []
    
    for object in raw_results:
        if object['class_name'] == 'person':
            person_list.append(object['id'])
    for object in action_out:
        action_list.append(object['id'])
        
    permut = itertools.permutations(person_list, len(action_list))
    
    for comb in permut:
        zipped = zip(comb, action_list)
        pairs.append(list(zipped))
    
    unique_comb = [{'assignments':x} for x in pairs]
    
    smallest_dist = 100000
    best_assign = []
    for y in unique_comb:
        sum_dist = 0
        for pair in y['assignments']:
            dist = math.dist(raw_results[int(pair[0])]['centroid'], action_out[int(pair[1])]['centroid'])
            sum_dist = sum_dist + dist
        if sum_dist < smallest_dist:
            smallest_dist = sum_dist
            best_assign = y['assignments']
    
    return best_assign

#-------

def action_assign(raw_results,action_out,best_assign):
    for pair in best_assign:
        raw_results[int(pair[0])].update({'action': action_out[int(pair[1])]['action']})
    
    return raw_results

#-------

def run_yolo_act(image, confidence, raw_results):
    action_out = standalone_yolo_act(image, confidence)
    best_assign = assign(raw_results, action_out)
    updated_results = action_assign(raw_results, action_out, best_assign)
    return updated_results

#------

def output_class_list_w_action(olist):
    labels = []
    for o in olist:
        if 'action' in o.keys():
            words = o['action'].split('_')
            labels.append(o['class_name']+' '+' '.join(words))
        else:
            labels.append(o['class_name'])
    return labels