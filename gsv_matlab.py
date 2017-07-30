
import h5py
import numpy as np

#takes file reader and reference of the type f[/digitStruct][name][INDEX] and returns image name
def readString(ref,f):
    d1=f[ref[0]]
    string=''.join(chr(alpha) for alpha in d1)
    return string

#takes file reader and reference of the type f[/digitStruct][bbox][INDEX] and returns bbox details
def readBbox(ref,f):
    d1=f[ref[0]]
    list=[]
    #d2 is a dataset of elements-height,width etc.
    #its size depends on the number of numbers in the given png
    for element in d1:
        list1=[]
        d2=d1[element]
        if(d2.shape==(1,1)):
            list1.append(d2[0][0])
            list.append(list1)

        else:
            for ref1 in d2:
                ref2=f[ref1[0]]
                list1.append(ref2[0][0])
            list.append(list1)
    return list


def extractor(dir):
    #needs to extract names and bbox lists and stores in numpy arrays
    #one array for names and one array for bbox
    f=h5py.File('/Users/Starck/Mark2/image_dir/'+dir+'/digitStruct.mat','r')
    g1=f['/digitStruct']
    dir_len=0
    if dir == 'train':
        dir_len=33402
    elif dir == 'test' :
        dir_len=13068
    elif dir == 'extra' :
        dir_len=30000
    names=np.zeros((dir_len,1),dtype=object)
    bbox=np.zeros((dir_len,1),dtype=object)
    # print(dir_len)
    for i in range(0,dir_len):
    # for i in range(0,3):
        ref1=g1['name'][i]
        name_string=readString(ref1,f)
        ref2=g1['bbox'][i]
        bbox_struct=readBbox(ref2,f)
        names[i][0]=name_string
        bbox[i][0]=bbox_struct
        if(i%1000==0):
            print(str(i)+'is being decoded from .mat file')
        
    return (names,bbox)
