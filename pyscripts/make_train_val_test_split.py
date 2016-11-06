import numpy as np
import shutil
import os
import glob
import sys
import random


def choose_image_only(fl):
    '''
    This function returns true if the file is a image file.
    '''
    ext = (".jpg", ".jpeg", ".JPG", ".png")
    return fl.endswith(ext)
def choose_random_img(s_lst, n_test = 0.1, n_val = 0.2):
    '''
    Randomly choose test images for each species
    
    s_lst : List of images of a particular species
    n_species : Percentage (specified as float) to be selected as test
    '''
    img_filenames = s_lst# filter(choose_image_only, s_lst)
    test_files = random.sample(img_filenames, int(n_test*len(s_lst)))
    temp_files = [x for x in img_filenames if x not in test_files]
    val_files = random.sample(temp_files, int(n_val*len(s_lst)))
    train_files = [x for x in temp_files if x not in val_files]
    return (train_files, val_files, test_files)

def make_img_path(lst):
    '''
    Function to make paths for images
    '''
    return map(lambda x: lst[0]+'/'+ x, lst[1])

def make_dirtree(src, dest):
    '''
    Function to copy folder structure without files
    '''
    cmd = 'rsync -a -f"+ */" -f"- *" %s/ %s/'%(src, dest)
    os.system(cmd)
    return None

def copy_images(tup):
    src,dst = tup
    shutil.copyfile(src, dst)
    return None

def make_train_test(folder):
    '''
    folder = parent_folder
    file system : *nix stored in the following structure
    parent folder(work/)
            |-- grayscale_consol/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/ 
            |            |- images.jpg
            |           .
            |           .
            |-- train/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg
            |           .
            |           . 
            |-- val/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg
            |           .
            |           .                          
            |-- test/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg        
            |           .
            |           .        
    '''

    # define folder paths
    
    # grayscale_consol : contains the grayscales of images in consol
    all_dat = folder + 'grayscale_consol/' #'consol/' #
    train = folder + 'g_train/'
    val = folder + 'g_val/'
    test = folder + 'g_test/'

    # number of species randomly selected 
    # split = test = 10%, val = 20%, train = 70%
    n_test = 0.1
    n_val = 0.2
    
    # Delete the directories if they ar already present.
    try:
        shutil.rmtree(train)
    except:
        pass    
    try:
        shutil.rmtree(val)
    except:
        pass 
    try:
        shutil.rmtree(test)
    except:
        pass 
    
    os.mkdir(train)
    os.mkdir(val)
    os.mkdir(test)
    
    # Create the directories of species without copying the files
    make_dirtree(all_dat, train)
    make_dirtree(all_dat, val)
    make_dirtree(all_dat, test)
    
    # get the species list stored in different folders. Filter out the hidden folders
    species_list = glob.glob(all_dat +'*')
    gen_path = map(os.walk, species_list)
    # nested list of path and the corresponding image filenames for every species
    lst_i = map(lambda x: x.next(), gen_path)
    # Converting the above list to array
    lst_a = np.array(lst_i, dtype = object)
    
    # Selecting the train, val, test images
    
    # Nested list of test image filenames for every class of species
    # first list is set of train images
    # second list is validation set
    # third is test set
    combined_list = map(lambda x: choose_random_img(x, n_test, n_val), lst_a[:,2])    
    combined_arr = np.array(combined_list)
    
    train_list = combined_arr[:,0]
    val_list = combined_arr[:,1]
    test_list = combined_arr[:,2]
    
    # Zipping the orginal dirdctory location of the images with the training images selected.
    train_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(train_list)))
    # Flattening the list
    train_src = reduce(lambda x,y: x+y,train_img_srcpaths)
    # Now chaning the directory location from source location to the training path
    train_src_dest = map(lambda x : (x,x.replace(all_dat, train)), train_src)    

    
    # Repeating the above process to get the validation directory path and the test path
    val_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(val_list)))
    val_src = reduce(lambda x,y: x+y,val_img_srcpaths)
    val_src_dest = map(lambda x : (x,x.replace(all_dat, val)), val_src)    

    
    test_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(test_list)))
    test_src = reduce(lambda x,y: x+y,test_img_srcpaths)
    test_src_dest = map(lambda x : (x,x.replace(all_dat, test)), test_src)
      
    # saving train and test images
    no_res = map(copy_images, train_src_dest)
    no_res1 = map(copy_images, test_src_dest)
    no_res2 = map(copy_images, val_src_dest)
    return None     

def main(work_folder):
    make_train_test(work_folder)

# Example to run this script 
# python <pathToFile>/make_train_test_split.py <pathToFolder>/work/

if __name__ == '__main__':
    work_folder = sys.argv[1]
    main(work_folder)
