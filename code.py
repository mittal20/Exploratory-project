from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.var_model import VAR
import pandas
import math
from sklearn.metrics import classification_report
import tsfresh

data=[]
with open("dataset3") as f:
  for line in f:
    l=list(map(float,line.split(",")))
    data.append(l)
x=[]
y=[]
for i in data:
  x.append(i[:-1])
  y.append(i[-1])

rows=len(x)
columns=len(x[0])

x=np.array(x)
y=np.array(y)
y=np.reshape(y,(len(y),1))


i=0
while i<len(x):
  flag=False
  for j in range(columns):
    if x[i][j]>10000 or x[i][j]<1000:
      flag=True
      break
  if flag==True:
    x=np.delete(x,i,0)
    y=np.delete(y,i,0)
  else:
    i+=1
rows=len(x)
columns=len(x[0])
x=np.array(x)
y=np.array(y)
y=np.reshape(y,(len(y),1))
temp=[range(rows)]
temp=np.array(temp)
temp=np.transpose(temp)
plt.plot(temp,x[:,0])
f1=plt.figure()

k=10
for i in range(columns):
  sum=0
  for j in range(k+1):
    sum+=x[j][i]
  for j in range(k,rows-1):
    x[j][i]=sum/k
    sum+=x[j+1][i]
    sum-=x[j-k][i]

plt.plot(temp,x[:,0])
f2=plt.figure()

b=griddata(x, y, x, method='nearest')
b=np.array(b)
for i in b:
  if i[0]>=0.5:
    i[0]=1
  else:
    i[0]=0

 

smtdata=x
smtdata=np.append(smtdata,y,axis=1)

train = smtdata[:int(0.7*(len(smtdata)))]
valid = smtdata[int(0.7*(len(smtdata))):]

model = VAR(endog=train)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=len(valid))

for i in yhat:
  if i[-1]>=0.5:
    i[-1]=1
  else:
    i[-1]=0

j=int(0.7*(len(smtdata)))

plt.show()

tp=0
tn=0
fp=0
fn=0

for i in range(len(yhat)):
  if yhat[i][14]>=0.5 and valid[i][0]>=0.5:
    tp+=1
  elif yhat[i][14]>=0.5 and valid[i][0]<0.5:
    fp+=1
  elif yhat[i][14]<0.5 and valid[i][0]>=0.5:
    fn+=1
  elif yhat[i][14]<0.5 and valid[i][0]<0.5:
    tn+=1

p=tp/(tp+fp)
r=tp/(tp+fn)
fs=2*p*r/(p+r)
accuracy=(tp+tn)/(tp+tn+fp+fn)
print("accuracy: ",accuracy)
print("f-score: ",fs)

x=np.insert(x,0,1,axis=1)
for i in range(rows):
  x[i][0]=i
y=np.insert(y,0,1,axis=1)

df=pandas.DataFrame(x,columns=['id',1,2,3,4,5,6,7,8,9,10,11,12,13,14])
df2=pandas.DataFrame(y,columns=['id',1])
# print(df2)
# print('aaaaaaa')


df3 = tsfresh.extract_features(df,column_id='id',default_fc_parameters = tsfresh.feature_extraction.MinimalFCParameters())

temporal=np.array(df3.values.tolist())
temporal=np.delete(temporal,1,axis=1)
temporal=np.delete(temporal,-1,axis=1)
rows=len(temporal)
columns=len(temporal[0])
print(rows,columns)
i=0
while i<len(temporal[0]):
  if temporal[0][i]==0 or temporal[0][i]==1:
    temporal=np.delete(temporal,i,1)
  else:
    i+=1
temporal=np.append(temporal,y,1)
columns=len(temporal[0])
print(rows,columns)

'''
print(len(temporal[0]))
f=open("abc.txt","w+")
for i in temporal:
  s=""
  for j in i:
    s+=str(j)
    s+=' '
  f.write(s)
f.close()
'''


def DTWDistance(s1, s2,w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return math.sqrt(DTW[len(s1)-1, len(s2)-1])

def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return math.sqrt(LB_sum)



def knn(train,test,w):
    preds=[]
    for ind,i in enumerate(test):
        min_dist=float('inf')
        closest_seq=[]
        for j in train:
            if LB_Keogh(i[:-1],j[:-1],5)<min_dist:
                dist=DTWDistance(i[:-1],j[:-1],w)
                if dist<min_dist:
                    min_dist=dist
                    closest_seq=j
        preds.append(closest_seq[-1])
    return classification_report(test[:,-1],preds) 

train = temporal[1000:2000]
test =  temporal[2000:2100]
print(knn(train,test,5))

data=np.vstack((train[:,:-1],test[:,:-1]))
data=list(data)
import random
def k_means_clust(data,num_clust,num_iter,w=5):
    centroids=random.sample(data,num_clust)
    counter=0
    for n in range(num_iter):
        counter+=1
        print(counter)
        assignments={}
        for ind,i in enumerate(data):
            min_dist=float('inf')
            closest_clust=None
            for c_ind,j in enumerate(centroids):
                if LB_Keogh(i,j,5)<min_dist:
                    cur_dist=DTWDistance(i,j,w)
                    if cur_dist<min_dist:
                        min_dist=cur_dist
                        closest_clust=c_ind
            if closest_clust in assignments:
                assignments[closest_clust].append(ind)
            else:
                assignments[closest_clust]=[]

        #recalculate centroids of clusters
        for key in assignments:
            clust_sum=0
            for k in assignments[key]:
                clust_sum=clust_sum+data[k]
            centroids[key]=[m/len(assignments[key]) for m in clust_sum]
    return centroids
centroids=k_means_clust(data,2,10,5)
for i in centroids:

    plt.plot(i)

plt.show()
