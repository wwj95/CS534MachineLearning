import numpy as np
import math
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/train-features.txt'
train_feature = np.loadtxt(filename)
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/test-features.txt'
text_feature = np.loadtxt(filename)
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/train-labels.txt'
file=open(filename, 'r')
train_label = file.read()
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/test-labels.txt'
file=open(filename, 'r')
text_label = file.read()
#print len(train_feature)
#print train_label
#print len(train_label)


classvote_0={}
classvote_1={}
for i in range(len(train_feature)):
    votelabel=train_feature[i][1]
    #print train_feature[i]
    #print train_label[int(train_feature[i+1][0])]
    if train_feature[i][0]<=350:

        if  votelabel in classvote_0:
            classvote_0[votelabel]+=train_feature[i][2]
        else:
            classvote_0[votelabel]=train_feature[i][2]
    if train_feature[i][0]>=350:

        if  votelabel in classvote_1:
            classvote_1[votelabel]+=train_feature[i][2]
        else:
            classvote_1[votelabel]=train_feature[i][2]

#print classvote_0,classvote_1

sum_0=0
sum_1=0
for i in range(len(train_feature)):
    if train_feature[i][0]<=350:
        sum_0+=train_feature[i][2]
    if train_feature[i][0]>=350:
        sum_1+=train_feature[i][2]
print sum_0, sum_1


def getsample(num):
    sample=[]
    for i in range(len(train_feature)):
        if train_feature[i][0]==num:
            sample.append(train_feature[i].tolist())
    return sample

def predict_for_spam():
    probability=0
    for line in sample:
        probability+=line[2]*math.log(classvote_0[line[1]]/sum_0)
    return probability
def predict_for_nonspam():
    probability=0
    for line in sample:
        probability+=line[2]*math.log(classvote_1[line[1]]/sum_1)
    return probability



sample=getsample(1)
p0=predict_for_spam()
p1=predict_for_nonspam()
#####################################################
import numpy as np
import math
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/train-features.txt'
train_feature = np.loadtxt(filename)
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/test-features.txt'
test_feature = np.loadtxt(filename)

filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/train-labels.txt'
file=open(filename, 'r')
train_label = file.read().split('\n')
train_label.pop()
filename='/Users/wendy/Documents/2017 Fall/CS 534/homework1/spam-dataset/test-labels.txt'
file=open(filename, 'r')
test_label = file.read().split('\n')
test_label.pop()


classvote_0={}
classvote_1={}
for i in range(len(train_feature)):
    votelabel=train_feature[i][1]
    if int(train_label[int(train_feature[i][0])-1])==0:
    #if train_feature[i][0]<=350:
        if  votelabel in classvote_0:
            classvote_0[votelabel]=classvote_0[votelabel]+train_feature[i][2]
        else:
            classvote_0[votelabel]=train_feature[i][2]
    if int(train_label[int(train_feature[i][0])-1])==1:
    #if train_feature[i][0]>350:

        if  votelabel in classvote_1:
            classvote_1[votelabel]=classvote_1[votelabel]+train_feature[i][2]
        else:
            classvote_1[votelabel]=train_feature[i][2]

#print classvote_1,classvote_0




sum_0=0
sum_1=0
for i in range(len(train_feature)):
    if train_feature[i][0]<=350:
        sum_0+=train_feature[i][2]
    if train_feature[i][0]>350:
        sum_1+=train_feature[i][2]
print sum_0, sum_1


def getsample(num):
    sample=[]
    for i in range(len(test_feature)-1):
        if test_feature[i][0]==num:
            sample.append(test_feature[i].tolist())
    return sample


def predict_for_spam():
    probability=0
    for line in sample:
        if line[1] in classvote_0:
            probability=probability+line[2]*math.log(classvote_0[line[1]]/sum_0)
            #print line[1],classvote_0[line[1]]
        else:
            probability=probability+line[2]*math.log(1/(sum_0+280))
    return probability
def predict_for_nonspam():
    probability=0
    for line in sample:
        if line[1] in classvote_1:
            probability=probability+line[2]*math.log(classvote_1[line[1]]/sum_1)
            #print line[1],classvote_0[line[1]]
        else:
            probability=probability+line[2]*math.log(1/(sum_1+280))
    return probability


#####err_rate for test###############
final_result=[]
p1_list=[]
for i in range(260):
    result=[i]
    sample=getsample(i)
    #print sample
    p0=predict_for_spam()
    p1=predict_for_nonspam()
    p1_list.append(p1)
    #print p1_list
    if p0>=p1:
        result.append('0')
    else:
        result.append('1')

    #if i<=130:
    if int(test_label[i])==0:
        result.append('0')
    else:
        result.append('1')
    final_result.append(result)
#print final_result
err=0
for i in final_result:
    if i[1]!=i[2]:
        err+=1
print'err_rate:',err

from sklearn.metrics import roc_auc_score
score=p1_list
label=[]
for i in test_label:
    i=int(i)
    label.append(i)
#print label
auc = roc_auc_score(label,score)
print 'AUC:',auc

def getsample_train(num):
    sample=[]
    for i in range(len(train_feature)-1):
        if train_feature[i][0]==num:
            sample.append(train_feature[i].tolist())
    return sample

###########err_rate for train############
p1_list=[]
final_result=[]
for i in range(700):
    result=[i]
    sample=getsample_train(i)
    p1_list.append(p1)
    #print sample
    p0=predict_for_spam()
    p1=predict_for_nonspam()
    if p0>=p1:
        result.append('0')
    else:
        result.append('1')

    if int(train_label[i-1])==0:
        result.append('0')
    else:
        result.append('1')
    final_result.append(result)
#print final_result
err=0
for i in final_result:
    if i[1]!=i[2]:
        err+=1
print'err_rate:',err

from sklearn.metrics import roc_auc_score
score=p1_list
label=[]
for i in train_label:
    i=int(i)
    label.append(i)
#print label
auc = roc_auc_score(label,score)
print 'AUC:',auc

       
