from mailbox import linesep
from tutorial import *
from auto import *
#PATH = "C:\\Users\\Geovanni\\Documents\\DATASETS\\chromedriver.exe"

#wd = webdriver.Chrome(PATH)

#urls = get_images_from_google(wd, 1, 120)

PATH = "C:\\Users\\Geovanni\\Documents\\DATASETS\\chromedriver.exe"

folders = open("fruits.txt",'r')

links = open("mass_links.txt",'r')
mkdir()

def multi():
    for l,f in zip(links,folders):
        sub_string = f.strip()
        
        #if sub_string.lower() in l:
            #print("ALREADY RAN")
        #else: 
        wd = webdriver.Chrome(PATH)
        images = get_images_from_google(wd,1,140,l)
        for i, image in enumerate(images):
            dest = "".join([f.strip(),"/"])
            download_image(dest,image, str(i) + ".jpg")
        wd.quit()

def single():
    counter = 1
    for l,f in zip(links,folders):
        wd = webdriver.Chrome(PATH)
        images = get_images_from_google(wd,1,100,l)
        for i,image in enumerate(images):
            download_image("MANGO/",image,str(counter) + ".jpg")
            counter = counter + 1
        wd.quit()

single()