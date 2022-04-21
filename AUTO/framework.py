from mailbox import linesep
from msilib.schema import Directory
from methods_fw import *


#Path to the chrome driver.
flag = True
img_counter = 0
set_flag = True
print("Welcome to the Data Scraper v0.1")
while(flag):
    
    direc_name = input("Enter New Directory name: ")
    flag_one = mkdir(direc_name)

    if(flag_one):
        while(set_flag):
            url = input("Enter a URL:")
            new_counter = single(direc_name,url,img_counter)
            img_counter = new_counter
            yes_no = input("Do you want to enter a new link (y/n):")
            if(yes_no == "y"):
                set_flag = True
            else:
                set_flag = False
                flag_one = False
                flag = False
    else:
        flag = True

print("Ending program")


    
    
    
    
