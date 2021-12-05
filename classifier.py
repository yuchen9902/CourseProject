import numpy as np
import math
from tqdm import tqdm
from collections import Counter
def count(train_set,train_labels,type):
    output = {}
    for i in range(len(train_labels)):
        if train_labels[i] == type:
            for word in train_set[i]:
                if word in output:
                    output[word] +=1
                else:
                    output[word]=1
    return output

def prob(Count_dict,laplace):
    prob_dict = {}
    totType = len(Count_dict)
    totWord = 0
    for i in Count_dict:
        totWord+=Count_dict[i]
    unk = laplace/(totWord+laplace*(totType+1))

    for i in Count_dict:
        p = (Count_dict[i]+laplace)/(totWord+laplace*(totType+1))
        prob_dict[i] = p
    
    return prob_dict,unk

def unigram(train_data,train_label,test_data):
    posCount_dict = count(train_data,train_label,1) 
    negCount_dict = count(train_data,train_label,0)

    laplace = 0.3
    pos_prior = 0.95

    posProb_dict,unk_pos = prob(posCount_dict,laplace)
    negProb_dict,unk_neg = prob(negCount_dict,laplace)

    yhats = []

#############################For running customize input#########################
    pos = 0
    neg = 0
    for word in test_data:
        if word in posProb_dict:
            pos += math.log(posProb_dict[word])
        else:
            pos += math.log(unk_pos)
        if word in negProb_dict:
            neg += math.log(negProb_dict[word])
        else:
            neg += math.log(unk_neg)
    pos +=  math.log(pos_prior)   
    neg += math.log(1-pos_prior)  
    if pos>neg:
        yhats.append(1)
    else:
        yhats.append(0)
    #############################For running test folder#########################
    # for doc in tqdm(test_data,disable=False):
    #     pos = 0
    #     neg = 0
    #     for word in doc:
    #         if word in posProb_dict:
    #             pos += math.log(posProb_dict[word])
    #         else:
    #             pos += math.log(unk_pos)
    #         if word in negProb_dict:
    #             neg += math.log(negProb_dict[word])
    #         else:
    #             neg += math.log(unk_neg)
    #     pos +=  math.log(pos_prior)   
    #     neg += math.log(1-pos_prior)  
    #     if pos>neg:
    #         yhats.append(1)
    #     else:
    #         yhats.append(0)
    
    return yhats

