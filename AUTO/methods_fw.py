import os
from pickle import FALSE, TRUE

from selenium import webdriver
from selenium.webdriver.common.by import By
from mailbox import linesep
from tutorial import *


PATH = "C:\\Users\\Geovanni\\Documents\\DATASETS\\FOLDERS\\chromedriver.exe"
main_directory =  "C:\\Users\\Geovanni\\Documents\\DATASETS\\FOLDERS\\"

def mkdir(name):
    name = name.strip()
    temp = "".join([name,"\\"])
    direc = "".join([main_directory,temp])
    print(direc)
    if os.path.exists(direc):
        print("ALREADY MADE: " + name)
        return False
    else:
        os.mkdir(os.path.join(main_directory,f"{name}"))
        print("Directory '% s' created" % name)
        return True

def single(name,url,counter):
    if(counter == 0):
        counter == 1
    else:
        counter = counter + 1

    wd = webdriver.Chrome(PATH)
    images = get_images_from_google(wd,1,10,url)
    for image in enumerate(images):
        download_image("mango/",image,str(counter) + ".jpg")
        counter = counter + 1
    wd.quit()

    return counter
