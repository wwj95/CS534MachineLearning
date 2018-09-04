import csv
import numpy
import scipy
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt



from numpy import *
import random
def SGD_Ridge(lam,learning_rate):
    lam=lam
    lr=learning_rate
    #threadhold = threadhold
    max_count=600
    w=mat(zeros((1,281)))
    training_list=[]
    validate_list=[]
    testdata_list=[]
    wholedata_list=[]
    tradata=csv.reader(open('/Users/wendy/Documents/2017 Fall/CS 534/homework1/BlogFeedback/blogData_train.csv'))
    for row in tradata:
        training_list.append(row)
    valdata=csv.reader(open('/Users/wendy/Documents/2017 Fall/CS 534/homework1/BlogFeedback/blogData_validate.csv'))
    for row in valdata:
        validate_list.append(row)
    testdata=csv.reader(open('/Users/wendy/Documents/2017 Fall/CS 534/homework1/BlogFeedback/blogData_test.csv'))
    for row in testdata:
        testdata_list.append(row)

    wholedata_list.extend(training_list)
    wholedata_list.extend(testdata_list)
    wholedata_list.extend(validate_list)
    for i in wholedata_list:
        i.insert(0,1)
    whole_list=numpy.array(wholedata_list,dtype=float)
    y = whole_list[:,-1]
    x = whole_list[:, 0:281]
    f=[]
    index=0
    while index < max_count:
        i=random.randint(0,len(y))
        xsample=mat(x[i])
        ysample=mat(y[i])
        gradient_w=xsample*xsample.T*w-ysample*xsample+lam*w
        w=w-lr*gradient_w
        index = index + 1
        f.append(float(w*xsample.T))
    #print f
    return (w,f)


#a=numpy.logspace(,0,10)
#for i in a:
w,f=SGD_Ridge(51607.487103859072,0.0000000001)
plt.plot(f,label="0.000000001")
plt.show()
