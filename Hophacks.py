
# coding: utf-8

# In[49]:


import scipy.io
import numpy as np
import pandas as pd
digit_train_mat = scipy.io.loadmat('digits-train.mat')
digit_test_mat = scipy.io.loadmat('digits-test.mat')


# In[17]:


digit_train_mat.keys()


# In[55]:


img_train_df = pd.DataFrame(digit_train_mat['images_train']).T
fea_hog_train_df = pd.DataFrame(digit_train_mat['fea_hog_train']).T
fea_scat_train_df = pd.DataFrame(digit_train_mat['fea_scat_train']).T
labels_train = digit_train_mat['labels_train']


# In[58]:


labels_true = np.empty([5000], dtype = int)
for i, label in enumerate(labels_train):
    labels_true[i] = label


# In[22]:


img_test_df = pd.DataFrame(digit_test_mat['images_test']).T
fea_hog_test_df = pd.DataFrame(digit_test_mat['fea_hog_test']).T
fea_scat_test_df = pd.DataFrame(digit_test_mat['fea_scat_test']).T


# In[26]:


img_train_df
fea_hog_train_df
fea_scat_train_df.shape
df_train = pd.concat([img_train_df, fea_hog_train_df, fea_scat_train_df], axis = 1)
df_train.shape


# In[28]:


df_test = pd.concat([img_test_df, fea_hog_test_df, fea_scat_test_df], axis = 1)


# In[61]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 5).fit(df_train)


# In[67]:


from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters = 5).fit(df_train)


# In[62]:


labels_pred = sc.predict(fea_scat_train_df)


# In[63]:



from sklearn.metrics.cluster import supervised
from scipy.optimize import linear_sum_assignment

labels_true, labels_pred = supervised.check_clusterings(labels_true, labels_pred)
# labels_true : int array with ground truth labels, shape = [n_samples]
# labels_pred : int array with estimated labels, shape = [n_samples]
value = supervised.contingency_matrix(labels_true, labels_pred)
# value : array of shape [n, n] whose (i, j)-th entry is the number of samples in true class i and in predicted class j
[r, c] = linear_sum_assignment(-value)
accr = value[r, c].sum() / len(labels_true)


# In[64]:


accr

