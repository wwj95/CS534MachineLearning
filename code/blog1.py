import csv
import numpy
import scipy
from sklearn import linear_model
from math import sqrt
import matplotlib.pyplot as plt
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

training_list=numpy.array(training_list,dtype=float)
validate_list=numpy.array(validate_list,dtype=float)
print len(training_list[0])
print len(training_list)

def Z_ScoreNormalization(x,mu,sigma):
    x = (x - mu) / sigma
    return x

for i in range(len(training_list[0])):
    mu=numpy.average(training_list[:,i])
    sigma=numpy.std(training_list[:,i])
    for j in range(len(training_list)):
        Z_ScoreNormalization(training_list[j,i],mu,sigma)



ytrain=training_list[:,280]
xtrain=training_list[:, 0:279]
yval=validate_list[:,280]
xval=validate_list[:,0:279]

lamlist=list(numpy.logspace(-2,5,1000))
RMSE=[]
for lam in lamlist:
    RidgeModel=linear_model.Ridge(lam)
    RidgeModel.fit(xtrain,ytrain)
    RMSE.append(numpy.linalg.norm((yval-RidgeModel.predict(xval)), 2)/sqrt(len(yval)))

min_rmse=0
for i in range(len(RMSE)):
    if RMSE[i]<min_rmse or min_rmse==0:
        min_rmse=RMSE[i]
        index=i
print (RMSE[index],lamlist[index])

plt.plot(lamlist,RMSE,'k')
plt.xlabel("-log(lambda)")
plt.xscale("log")
plt.ylabel('RMSE')
plt.show()
