import numpy as np
import cv2

def extract_patches(input_array, patch_shape, stride):
    """
    Extract patches from a 2D NumPy array.
    
    Parameters:
    - input_array (np.array): The input 2D array.
    - patch_shape (tuple): The shape of the patches (rows, columns).
    - stride (tuple): The stride for patch extraction (row_stride, col_stride).
    
    Returns:
    - List[np.array]: List of extracted patches.
    """
    patches = []
    rows, cols = input_array.shape
    patch_rows, patch_cols = patch_shape
    row_stride, col_stride = stride
    
    for r in range(0, rows - patch_rows + 1, row_stride):
        for c in range(0, cols - patch_cols + 1, col_stride):
            patch = input_array[r:r + patch_rows, c:c + patch_cols]
            patches.append(patch)
    
    return patches

def my_division_data(shape=(992,576),stridetest=(230,14), strideval=(230,14), stridetrain=(8,8)):
    read_seis_data = np.load('/scratch/nuneslima/seismic_data/data_train.npz', 
                allow_pickle=True, mmap_mode = 'r')
    # We read our labels
    read_labels = np.load('/scratch/nuneslima/seismic_data/labels_train.npz',
                    allow_pickle=True, mmap_mode = 'r')

    # Inside the elements we pick what we are interesed in
    seis_data = read_seis_data['data']
    labels = read_labels['labels']
    labels[labels==6] = 0


    testcrossline=seis_data[:,702:782,:]
    testinline=seis_data[:,:702,510:590]
    testcrossline_label=labels[:,702:782,:]
    testinline_label=labels[:,:702,510:590]


    #removing the test data the rest of our data has shape Z=1006 X=702 Y=510
    valcrossline=seis_data[:,622:702,:510]
    valinline=seis_data[:,:622,430:510]
    valcrossline_label=labels[:,622:702,:510]
    valinline_label=labels[:,:622,430:510]


    ##removing the validation data the rest of our data has shape Z=1006 X=622 Y=430
    traindata=seis_data[:,:622,:430]
    trainlabel=labels[:,:622,:430]


    #TRAINING
    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))
    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)

    #VALIDATION
    valpatches=[]
    vallabels=[]
    for i in (range(valinline.shape[2])):
        valpatches=valpatches+extract_patches(valinline[:,:,i],(shape),(strideval))
        vallabels=vallabels+extract_patches(valinline_label[:,:,i],(shape),(strideval))
    for i in (range(valcrossline.shape[1])):
        valpatches=valpatches+extract_patches(valcrossline[:,i,:],(shape),(strideval))
        vallabels=vallabels+extract_patches(valcrossline_label[:,i,:],(shape),(strideval))
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    #TEST
    testpatches=[]
    testlabels=[]
    for i in (range(testinline.shape[2])):
        testpatches=testpatches+extract_patches(testinline[:,:,i],(shape),(stridetest))
        testlabels=testlabels+extract_patches(testinline_label[:,:,i],(shape),(stridetest))
    for i in (range(testcrossline.shape[1])):
        testpatches=testpatches+extract_patches(testcrossline[:,i,:],(shape),(stridetest))
        testlabels=testlabels+extract_patches(testcrossline_label[:,i,:],(shape),(stridetest))
    testslices=np.array(testpatches)
    testlabels=np.array(testlabels)

    return trainslices, trainlabels, testslices, testlabels, valslices, vallabels




def article_division_data(shape=(992,576), strideval=(230,14), stridetrain=(8,8)):
    read_seis_data = np.load('/scratch/nuneslima/seismic_data/data_train.npz', 
            allow_pickle=True, mmap_mode = 'r')
    # We read our labels
    read_labels = np.load('/scratch/nuneslima/seismic_data/labels_train.npz',
                    allow_pickle=True, mmap_mode = 'r')

    # Inside the elements we pick what we are interesed in
    seis_data = read_seis_data['data']
    labels = read_labels['labels']
    labels[labels==6] = 0


    valinline=np.append(seis_data[:,:,:7],seis_data[:,:,584:], axis=2)
    valinline_label=np.append(labels[:,:,:7],labels[:,:,584:], axis=2)
    valcrossline=np.append(seis_data[:,:45,:],seis_data[:,738:,:], axis=1)
    valcrossline_label=np.append(labels[:,:45,:],labels[:,738:,:], axis=1)
    traindata=seis_data[:,45:738,7:583]
    trainlabel=labels[:,45:738,7:583]

    trainpatches=[]
    trainlabels=[]
    for i in (range(traindata.shape[2])):
        trainpatches=trainpatches+extract_patches(traindata[:,:,i],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,:,i],(shape),(stridetrain))
    for i in (range(traindata.shape[1])):
        trainpatches=trainpatches+extract_patches(traindata[:,i,:],(shape),(stridetrain))
        trainlabels=trainlabels+extract_patches(trainlabel[:,i,:],(shape),(stridetrain))

    valpatches=[]
    vallabels=[]
    for i in (range(valinline.shape[2])):
        valpatches=valpatches+extract_patches(valinline[:,:,i],(shape),(strideval))
        vallabels=vallabels+extract_patches(valinline_label[:,:,i],(shape),(strideval))
    for i in (range(valcrossline.shape[1])):
        valpatches=valpatches+extract_patches(valcrossline[:,i,:],(shape),(strideval))
        vallabels=vallabels+extract_patches(valcrossline_label[:,i,:],(shape),(strideval))
    
    trainslices=np.array(trainpatches)
    trainlabels=np.array(trainlabels)
    valslices=np.array(valpatches)
    vallabels=np.array(vallabels)

    return trainslices, trainlabels, valslices, vallabels