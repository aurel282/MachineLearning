from sklearn.cluster import KMeans

import numpy as np
dataset = np.loadtxt("heart.txt", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:13]
Y = dataset[:, 13]
XTest1 = dataset[0:91, 0:13]
YTest1 = dataset[0:91, 13]
XTest1 = np.concatenate([XTest1, dataset[181:271, 0:13]])
YTest1 = np.concatenate([YTest1, dataset[181:271, 13]])
XTest1Valid = dataset[91:181, 0:13]
YTest1Valid = dataset[91:181, 13]

kmeans = KMeans(n_clusters=2, random_state=0, algorithm='full')
kmeans.fit(XTest1,YTest1)

print(kmeans.cluster_centers_)
print(YTest1)
xx = kmeans.predict(XTest1Valid)

count = 0
pos = 0
for i in xx:
    if i == YTest1Valid[pos]:
        count += 1
    pos += 1

print(count / YTest1Valid.size * 100)
