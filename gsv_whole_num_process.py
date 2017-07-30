import tensorflow as tf
import os
import matplotlib
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
import gsv_matlab 

def normalize_img(img):
    np_img=np.array(img)
    im=np.dot(np_img,[0.299,0.587,0.114])
    mean = np.mean(im, dtype='float32')
    std = np.std(im, dtype='float32', ddof=1)
    return (im - mean) / std  

def process_imgs(dir,dir_len):
    new_img_list=[]
    new_img_labels=[]
    sum=0
    print('processing '+dir+' data') 
    names, bbox = gsv_matlab.extractor(dir)
    for j in range(1, dir_len+1):
        file='/Users/Starck/Mark2/image_dir/'+dir+'/'+str(j)+'.png'
        im=Image.open(file)
        i=j-1
        length=len(bbox[i][0][0])
        sum+=length
        if(length<4):
            height=bbox[i][0][0]
            label=bbox[i][0][1]
            left=bbox[i][0][2]
            top=bbox[i][0][3]
            width=bbox[i][0][4]
            '''im.crop(box) ⇒ image
            Returns a copy of a rectangular region from the current image.
            The box is a 4-tuple defining the left, upper, right, and lower pixel coordinate.'''
            left_cord=np.amin(left)
            right_cord=left_cord+np.sum(width)
            upper_cord=np.amin(top)
            lower_cord=upper_cord+np.amax(height)
            cord_tuple=(left_cord,upper_cord,right_cord,lower_cord)
            cropped_img=im.crop(cord_tuple)
            resized_img=cropped_img.resize((32,32))
            normalized_img=normalize_img(resized_img)
            new_img_list.append(normalized_img)
            new_img_labels.append(label)

        if(j%1000==0):
            print(str(j)+'th image has been processed')
    return new_img_labels,new_img_list 

def create_datasets():
    test_labels,test_images=process_imgs('test',13068)
    print('test data created....')
    train_labels,train_images=process_imgs('train',33402)
    print('train_data created...')
    extra_labels,extra_images=process_imgs('extra',30000)
    print('extra data created')
    # An arbitrary collection of objects supported by pickle.
    data = {
        'train_data': [train_images,train_labels],
        'test_data': [test_images,test_labels],
        'extra_data': [extra_images,extra_labels]
    }
    print('test:'+str(len(test_labels)))
    print('train:'+str(len(train_labels)))
    print('extra:'+str(len(extra_labels)))
    with open('whole_data.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        #since the data has been written in raw binary form<wb> it must be read as <rb> while unpickling
        print('pickling data...')
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data pickled...')


if __name__ == '__main__':
  create_datasets()
