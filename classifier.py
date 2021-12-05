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

####helper for bigram####
def bicount(train_set,train_labels,type):
    output = {}
    for i in range(len(train_labels)):
        if train_labels[i] == type:
            input = train_set[i]
            for j in range(len(input)-1):
                pair = (input[j],input[j+1])
                if pair in output:
                    output[pair] +=1
                else:
                    output[pair]=1
    return output

def biprob(Count_dict,train_set,laplace):
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

def bigram(train_set, train_labels, dev_set):
    unigram_laplace = 0.01
    bigram_laplace = 0.01
    bigram_lambda = 0.5
    pos_prior = 0.8
    posCount_dict = count(train_set,train_labels,1) 
    negCount_dict = count(train_set,train_labels,0)
    posProb_dict,unk_pos = prob(posCount_dict,unigram_laplace)
    negProb_dict,unk_neg = prob(negCount_dict,unigram_laplace)

    posOutput_uni = []
    negOutput_uni = []
    for doc in tqdm(dev_set,disable=False):
        pos = 0
        neg = 0
        for word in doc:
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
        posOutput_uni.append(pos)
        negOutput_uni.append(neg) 
    
    #bigram
    biPosCount_dict = bicount(train_set,train_labels,1) 
    biNegCount_dict = bicount(train_set,train_labels,0)
    biPosProb_dict,unk_pos_bi = biprob(biPosCount_dict,train_set,bigram_laplace)
    biNegProb_dict,unk_neg_bi = biprob(biNegCount_dict,train_set,bigram_laplace)
    posOutput_bi = []
    negOutput_bi = []
    for doc in tqdm(dev_set,disable=False):
        pos = 0
        neg = 0
        for j in range(len(doc)-1):
            word = (doc[j],doc[j+1])
            if word in biPosProb_dict:
                pos += math.log(biPosProb_dict[word])
            else:
                pos += math.log(unk_pos_bi)
            if word in biNegProb_dict:
                neg += math.log(biNegProb_dict[word])
            else:
                neg += math.log(unk_neg_bi)
        pos +=  math.log(pos_prior)   
        neg += math.log(1-pos_prior) 
        posOutput_bi.append(pos)
        negOutput_bi.append(neg) 


    yhats = []
    for i in range(len(posOutput_bi)):
        calc_pos = (1-bigram_lambda)*posOutput_uni[i] + bigram_lambda*posOutput_bi[i]
        calc_neg = (1-bigram_lambda)*negOutput_uni[i] + bigram_lambda*negOutput_bi[i]
        if calc_pos > calc_neg:
            yhats.append(1)
        else:
            yhats.append(0)
        
    return yhats