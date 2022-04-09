from pickle import FALSE, TRUE
from re import L
import cv2 as cv
from torch import maximum

import data_collection as dc
orb = cv.ORB_create()

list_names = ["apple","banana","blueberry","lemon","lime"]

def BF_FeatureMatcher(des1,des2):
	brute_force = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
	no_of_matches = brute_force.match(des1,des2)

	# finding the humming distance of the matches and sorting them
	no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)
	return int(len(no_of_matches))
def matcher(des1,des2):
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    result = int(len(good))
    return result



    
train_dict = dc.train_test_set_images(TRUE)
key_list_train = list(train_dict)
test_dict = dc.train_test_set_images(FALSE)
key_list_test = list(test_dict)

prediction_dict = {"APPLES" : None, "BANANA" : None, "BLUEBERRY" : None, "LEMON" :None, "LIME" : None}

#for item in train_dict["APPLES"]:
#    print(item)
#    print('-------')

def finding_label(img_test):
    maximum = 0
    maximum_two = 0
    title_label = "Empty"
    title_label_two = "Empty"
    for key in key_list_train:
        label = key.lower()
        for img_train in train_dict[f'{key}']:
            #test_desc = img_test[0][0]
            #train_desc = img_train[0][0]
            kp1, des1 = orb.detectAndCompute(img_test,None)
            kp2, des2 = orb.detectAndCompute(img_train,None)
            num_of_matches = BF_FeatureMatcher(des1,des2)
            num_of_matches_two = matcher(des1,des2)
            if maximum < num_of_matches:
                maximum = num_of_matches
                title_label = label
            if maximum_two < num_of_matches_two:
                maximum_two = num_of_matches_two
                title_label_two = label

    return title_label, title_label_two


print("PROCESS STARTING")
for key in key_list_train:
    print(f'<==={key}===>')
    accuracy_list = []
    correct_num = 0
    correct_num_two = 0
    total_num = 0

    
    

    for img_test in test_dict[f'{key}_TEST']:
        total_num = total_num + 1
        title_label, title_label_two = finding_label(img_test)
        
        accuracy_list.append(title_label)
        
        if key.lower() == title_label:
            correct_num = correct_num + 1
        if key.lower() == title_label_two:
            correct_num_two = correct_num_two + 1

        percentage = (correct_num/total_num) * 100
        percentage_two = (correct_num_two/total_num) * 100
        
        print(f'Image: {key.lower()} | Prediction(BF_FM): {title_label} | Accuracy(BF_FM) : {percentage} | Prediction(User Defined) : {title_label_two} | Accuracy(User Defined) : {percentage_two}')
    prediction_dict.update({f'{key}' : accuracy_list})
print("PROCESS END")

#for key in key_list_test:
#    test_list = test_dict.get(f'{key}')
#    kp_list = test_list[0][0]
#    des_list = test_list[0][1]
#    print(key)
#    print(len(test_list))
#    print(len(test_list[0]))
#    print(len(kp_list))
#    print(len(des_list))
#    print("-------")
