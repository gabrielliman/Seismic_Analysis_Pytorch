import cv2
import torch
import numpy as np
from tqdm import tqdm
import argparse
from unet import UNet
from tabulate import tabulate
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.datapreparation import my_division_data, article_division_data

def seisfacies_predict(model, section,patch_size=256,overlap=0,onehot=0): 
    m1,m2 = section.shape
    os    = overlap                                 
    n1,n2 = 512,patch_size           
    c1 = int(np.round((m1+os)/(n1-os)+0.5))
    c2 = int(np.round((m2+os)/(n2-os)+0.5))
    p1 = (n1-os)*c1+os
    p2 = (n2-os)*c2+os

    gp = np.zeros((p1,p2),dtype=np.single)     
    gy = np.zeros((6,p1,p2),dtype=np.single)    
    gs = np.zeros((n1,n2),dtype=np.single) 
    
    gp[0:m1,0:m2]=section     

    for k1 in range(c1):
        for k2 in range(c2):
            b1 = k1*n1-k1*os
            e1 = b1+n1
            b2 = k2*n2-k2*os
            e2 = b2+n2                
            #predict
            gs[:,:]=gp[b1:e1,b2:e2]
            x=gs.reshape(1,1,512,256)
            Y_patch= model((torch.from_numpy(x)).to('cuda')).squeeze()
            p=F.softmax(Y_patch.cpu(), dim=0).detach().numpy()
            gy[:,b1:e1,b2:e2]= gy[:,b1:e1,b2:e2]+p
    
    gy_onehot = gy[:,0:m1,0:m2]            
    #onehot2label
    gy_label =np.argmax(gy_onehot,axis=0)

    if onehot==0:
        return gy_label
    if onehot==1:
        return gy_label,gy_onehot
    
def calculate_accuracy(model, test_data):
    """
    Calculate the accuracy of the model on the given test data.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        test_data (list): A list of tuples, each containing a 2-dimensional test sample and its corresponding label.

    Returns:
        float: The accuracy of the model on the test data.
    """
    total_correct = 0
    total_samples = len(test_data)

    # Set the model to evaluation mode (important for models with dropout, batchnorm, etc.)
    model.eval()

    with torch.no_grad():
        for sample, label in tqdm(test_data):
            # Assuming 'sample' and 'label' are already in tensor format
            prediction = seisfacies_predict(sample)

            # If your prediction function returns probabilities, you can convert them to labels
            # predicted_label = torch.argmax(prediction, dim=1)

            # Compare the predicted label with the true label
            correct = (prediction == label).sum().item()
            total_correct += correct

    # Calculate accuracy
    accuracy = total_correct / (total_samples)


    return accuracy/(256*512)

def calculate_micro_f1_score(true_positives, false_positives, false_negatives):
    """
    Calculate the micro F1 score from true positives, false positives, and false negatives.

    Args:
        true_positives (int): Total true positives across all classes.
        false_positives (int): Total false positives across all classes.
        false_negatives (int): Total false negatives across all classes.

    Returns:
        float: The micro F1 score.
    """
    micro_precision = true_positives / (true_positives + false_positives)
    micro_recall = true_positives / (true_positives + false_negatives)
    micro_f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)

    return micro_f1_score

def calculate_class_info(model, test_data, num_classes):
    """
    Calculate the pixel-wise accuracy, precision, and recall for each class of the model on the given test data.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        test_data (list): A list of tuples, each containing a 2-dimensional test sample (image) and its corresponding label (image).
        num_classes (int): The number of classes in the dataset.

    Returns:
        dict: A dictionary containing pixel-wise accuracy, precision, and recall for each class.
    """
    # Initialize counters for true positives, false positives, and false negatives for each class
    true_positives = [0] * num_classes
    false_positives = [0] * num_classes
    false_negatives = [0] * num_classes

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        for sample, label in tqdm(test_data):
            # Assuming 'sample' and 'label' are already in tensor format
            predicted_label = seisfacies_predict(model,sample)

            for class_idx in range(num_classes):
                true_positive_mask = (predicted_label == class_idx) & (label == class_idx)
                false_positive_mask = (predicted_label == class_idx) & (label != class_idx)
                false_negative_mask = (predicted_label != class_idx) & (label == class_idx)

                true_positives[class_idx] += true_positive_mask.sum().item()
                false_positives[class_idx] += false_positive_mask.sum().item()
                false_negatives[class_idx] += false_negative_mask.sum().item()

    # Calculate pixel-wise accuracy, precision, and recall for each class
    accuracy_by_class = {}
    for class_idx in range(num_classes):
        total_pixels = true_positives[class_idx] + false_positives[class_idx] + false_negatives[class_idx]
        if true_positives[class_idx]!=0:
            accuracy_by_class[class_idx] = true_positives[class_idx] / total_pixels
            precision = true_positives[class_idx] / (true_positives[class_idx] + false_positives[class_idx])
            recall = true_positives[class_idx] / (true_positives[class_idx] + false_negatives[class_idx])
        else:
            accuracy_by_class[class_idx]=0
            precision=0
            recall=0
        accuracy_by_class[class_idx] = [
            accuracy_by_class[class_idx],
            precision,
            recall
        ]

     # Calculate total true positives, false positives, and false negatives across all classes
    total_true_positives = sum(true_positives)
    total_false_positives = sum(false_positives)
    total_false_negatives = sum(false_negatives)

    # Calculate micro F1 score
    micro_f1_score = calculate_micro_f1_score(total_true_positives, total_false_positives, total_false_negatives)

    return accuracy_by_class, micro_f1_score

def calculate_macro_f1_score(class_info):
    class_f1 = [0,1,2,3,4,5]
    for class_idx in class_info:
        precision = class_info[class_idx][1]
        recall = class_info[class_idx][2]
        if(precision+recall!=0):
            class_f1[class_idx] = 2 * (precision * recall) / (precision + recall)
        else:
            class_f1[class_idx]=0

    # Calculate the macro F1 score as the unweighted average of per-class F1 scores
    macro_f1_score = np.mean(list(class_f1))

    return macro_f1_score, class_f1

def prediciton(name, article_split=False,slice_shape1=992, slice_shape2=64):
    model = torch.load('/scratch/nuneslima/models/'+name+'.pth')
    model.eval()
    if(article_split==False):
        train_image, train_label, test_image, test_label, val_image, val_label=my_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(230,64), strideval=(230,64), stridetest=(230,64))
    else:
        train_image, train_label, test_image, test_label=article_division_data(shape=(slice_shape1,slice_shape2), stridetrain=(230,64), strideval=(230,64))

    test_data = []
    for i in range(len(test_image)):
        sample = test_image[i]
        label = test_label[i]
        test_data.append((sample, label))

    class_info, micro_f1=calculate_class_info(model, test_data, 6)
    macro_f1, class_f1=calculate_macro_f1_score(class_info)

    data=[]
    for i in range(len(class_info)):
        data.append(["Classe "+str(i)] + class_info[i]+[class_f1[i]])

  
    #define header names
    col_names = ["Classe","Accuracy", "Precision", 'Recall', 'F1 score']

    f = open("tables/table_"+name+".txt", "w")
    f.write(tabulate(data, headers=col_names, tablefmt="fancy_grid",floatfmt=".4f"))
    texto="\nMacro F1 "+ str(macro_f1) + "\nMicro F1 " + str(micro_f1)
    f.write(texto)
    f.close()

    
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--name', '-f', type=str, default=False, help='Model name for saving')
    parser.add_argument('--split', action='store_true', default=False, help='Use article data split')
    parser.add_argument('--slice_shape1', '-s1',dest='slice_shape1', metavar='S', type=int,default=992, help='Shape 1 of the slices used in training and validation')
    parser.add_argument('--slice_shape2', '-s2',dest='slice_shape2', metavar='S', type=int,default=64, help='Shape 2 of the slices used in training and validation')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    prediciton(name=args.name,
            article_split=args.split,
            slice_shape1=args.slice_shape1,
            slice_shape2=args.slice_shape2)