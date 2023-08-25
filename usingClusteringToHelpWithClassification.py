"""
Authors: Michael Baosv, Shlomo Gulayev, Micha Briskman
ID: 315223156, 318757382, 208674713
"""
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from kneed import KneeLocator
from sklearn.metrics import classification_report, confusion_matrix

#%%
def find_k(indices, digit):
    """Find the optimum k clusters"""
    sse = {}
    
    for k in range(1, 60):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=3072, n_init='auto')
        kmeans.fit(X_train[indices])
        sse[k] = kmeans.inertia_
        
    plt.title(f'Elbow plot for K selection for {digit}')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.show()
    
    kn = KneeLocator(x=list(sse.keys()), 
                 y=list(sse.values()), 
                 curve='convex', 
                 direction='decreasing')
    k = kn.knee
    return k

#%%
# Load data from https://www.openml.org/d/554
X, y = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="pandas")
X = X / 255.0

# Split data into train partition and test partition
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.5)

# identify all digits indices and find optimum clusters
cluster_num = 0
digits_indices = {}
for i in range(10):
    digits_indices[i] = np.where(y_train == str(i))
    cluster_num += find_k(digits_indices[i], i)

print(cluster_num)
#%%
kmeans = MiniBatchKMeans(n_clusters=3*cluster_num, random_state=0, batch_size=3072)
kmeans.fit(X_train)
prediction = kmeans.predict(X_test)
#%%
y_train = y_train.astype('int64') # convert strings to int
cluster_true = {}

for label in range(kmeans.n_clusters):
    indices = np.where(kmeans.labels_ == label)
    cluster_members = y_train[indices]
    
    frequency = np.bincount(np.squeeze(cluster_members))
    most_freq = frequency.argmax()

    if most_freq not in cluster_true:
        cluster_true[most_freq] = [label]
    else:
        cluster_true[most_freq].append(label)

prediction_labels = []
for pre in prediction:
    for key, value in cluster_true.items(): #find the label of the cluster
        if pre in value:
            prediction_labels.append(key)

y_test = y_test.astype('int64') # convert string to int
print(classification_report(y_test, prediction_labels))
print(confusion_matrix(y_test, prediction_labels))
