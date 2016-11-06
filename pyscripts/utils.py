# To check the count of the images in train, val,  test folders.
def choose_image_only(fl):
    '''
    This function returns true if the file is a png file.
    '''
    ext = (".jpg", ".jpeg", ".JPG", ".png")
    return fl.endswith(ext)


def img_count(root_dir):
    '''
    This function is used 
    '''
    img_lst = []
    di = os.walk(root_dir)
    try:
        while True:
            try:
                path, dirs, files = di.next()
                # Choose only the png files
                img_filenames = filter(choose_image_only, files)
                if img_filenames:
                    #print 'here'
                    img_lst.extend(img_filenames)
                    #temp_lst = map(lambda x: move_an_image(x, dest), img_lst )
                    #img_size_lst.extend(temp_lst)
            except StopIteration:
                break
        return img_lst
    except Exception as e:
        print "Error !::", str(e)
        

if __name__ == '__main__':
	train_count = img_count('./work/g_train/')
	val_count = img_count('./work/g_val/')
	test_count = img_count('./work/g_test/')

