from pickle import FALSE, TRUE
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import glob





def train_test_set(flag):
    # T -> TRAIN
    # F -> TEST

    #Dictionary
    #|->LIST
    #   |-> kp
    #       |-> specific value of kp
    #   |-> des
    #       |-> specific value of des

    orb = cv.ORB_create()
    width = 400
    height = 400
    dim = (width, height)

    list_of_sub_folders = []
    extend_str = None
    index = 0

    if flag == TRUE:
        dict = { "APPLES" : None, "BANANA" : None, "BLUEBERRY" : None, "LEMON" :None, "LIME" : None}
        extend_str = 'PARENT_TRAIN'
    elif flag == FALSE:
        dict = { "APPLES_TEST" : None, "BANANA_TEST" : None, "BLUEBERRY_TEST" : None, "LEMON_TEST" :None, "LIME_TEST" : None}
        extend_str = 'PARENT_TEST'
    #print(lime_dict["LIME"][2])
    keys_list = list(dict)

    rootdir = 'C:\\Users\\Geovanni\\Documents\\SIFT\\'
    for path in glob.glob(f'{rootdir}/{extend_str}/*/'):
        list_of_sub_folders.append(path)
    for sub in list_of_sub_folders:
        list_list = []    
        images_names = glob.glob(f'{sub}//*.jpg')
        images = [cv.resize(cv.imread(img, cv.IMREAD_GRAYSCALE),dim,interpolation = cv.INTER_AREA) for img in images_names]
        for img in images:
            kp,des = orb.detectAndCompute(img,None)
            tup = (kp,des)
            list_list.append(tup)
        key = keys_list[index]
        dict.update({f'{key}': list_list})
        #print(len(list_list))
        index = index + 1
    return dict

def train_test_set_images(flag):
    # T -> TRAIN
    # F -> TEST

    #Dictionary
    #|->LIST
    #   |-> kp
    #       |-> specific value of kp
    #   |-> des
    #       |-> specific value of des

    orb = cv.ORB_create()
    width = 600
    height = 600
    dim = (width, height)

    list_of_sub_folders = []
    extend_str = None
    index = 0

    if flag == TRUE:
        dict = { "APPLES" : None, "BANANA" : None, "BLUEBERRY" : None, "LEMON" :None, "LIME" : None}
        extend_str = 'PARENT_TRAIN'
    elif flag == FALSE:
        dict = { "APPLES_TEST" : None, "BANANA_TEST" : None, "BLUEBERRY_TEST" : None, "LEMON_TEST" :None, "LIME_TEST" : None}
        extend_str = 'PARENT_TEST'
    #print(lime_dict["LIME"][2])
    keys_list = list(dict)

    rootdir = 'C:\\Users\\Geovanni\\Documents\\SIFT\\'
    for path in glob.glob(f'{rootdir}/{extend_str}/*/'):
        list_of_sub_folders.append(path)
    for sub in list_of_sub_folders:
        list_list = []    
        images_names = glob.glob(f'{sub}//*.jpg')
        images = [cv.resize(cv.imread(img, cv.IMREAD_GRAYSCALE),dim,interpolation = cv.INTER_AREA) for img in images_names]
        key = keys_list[index]
        dict.update({f'{key}': images})
        #print(len(list_list))
        index = index + 1
    return dict






#images_names = glob.glob("C:\\Users\\Geovanni\\Documents\\SIFT\\LIME_TRAIN/*.jpg")
#images = [cv.resize(cv.imread(img, cv.IMREAD_GRAYSCALE),dim,interpolation = cv.INTER_AREA) for img in images_names]
#for img in images:
#    kp,des = orb.detectAndCompute(img,None)
#    tup = (kp,des)
    #print(tup)
#    list.append(tup)

#print(len(lime_dict["LIME"][1][0]))