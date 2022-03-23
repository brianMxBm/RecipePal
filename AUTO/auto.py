#from tutorial import *
import os
from re import search

folders = open("fruits.txt",'r')

links = open("fruits_links.txt",'r')
main_directory =  r'C:\Users\Geovanni\Documents\DATASETS'

def mkdir():
    for line in folders:
        line = line.strip()
        line_two = "".join(["/",line])
        newline = "".join([main_directory,line_two])
        print(newline)
        if os.path.exists(newline):
            print("ALREADY MADE: " + line)
        else:
            #print("NOT MADE YET: " + line)
            os.mkdir(os.path.join(main_directory,f"{line}"))
            print("Directory '% s' created" % line)
def comparing():
    for f,l in zip(folders,links):
        sub_string = f.strip()
        if sub_string.lower() in l:
            print("FOUND")
        else:
            print("NOT FOUND")
