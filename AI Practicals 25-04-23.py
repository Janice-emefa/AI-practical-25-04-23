#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


df = pd.read_csv("iris.csv")


# Data Analysis

# a. Load the iris dataset into a Pandas data frame and display the first five rows.

# In[18]:


df.head(5)


# b. Display the shape pf the dataset (number of rows and columns)

# In[19]:


df.shape


# c. Display the summary statistics (mean, standard deviation, minimum and maximum) for each column

# In[20]:


df.describe()


# d. Create a new column in the data frame that is the product of the Sepal length and Sepal Width columns

# In[21]:


#Create a new column
df["sepal.area"] = df["sepal.length"] * df["sepal.width"]
df.head()


# e. Display the unique values in the species column

# In[22]:


df["variety"].unique()


# f. Create a scatterplot ofthe sepal length vs. sepal width columns, with different colours for each species.

# In[25]:


#plotting a scatterplot
sb.scatterplot(data=df, x="sepal.length", y="sepal.width", hue="variety")
plt.plot()


# Machine Learning

# a. Split the dataset inti training and testing sets (70% training, 30% testing)

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Iris dataset
iris = pd.read_csv('iris.csv')

# Prepare the data
X = iris[['sepal.length', 'sepal.width']]
y = iris['variety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# b. Implement a logistic regression model to predict the species based on the sepal length and sepal width columns. Evaluate the model using the accuracy, precision, recall, and the F1 score.

# In[30]:


# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 score: ", f1_score(y_test, y_pred, average='macro'))


# c. Implement a k-nearest neighbours classifier with k = 3 to predict the species based on all four columns. Evaluate the model using accuracy, precision, recall, and the F1 score. 

# In[31]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# In[33]:


# K-nearest neighbors classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("K-Nearest Neighbors Classifier:")
print("Accuracy: ", accuracy_score(y_test, y_pred_knn))
print("Precision: ", precision_score(y_test, y_pred, average='macro'))
print("Recall: ", recall_score(y_test, y_pred, average='macro'))
print("F1 score: ", f1_score(y_test, y_pred, average='macro'))


# e. Implement a support vector machine (SVM) classifier to predict the species based on all four columns. Evaluate the model using accuracy, precision, recall, and the F1 score.

# In[35]:


from sklearn.svm import SVC


# In[36]:


# Support Vector Machine (SVM) classifier
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine (SVM) Classifier:")
print("Accuracy: ", accuracy_score(y_test, y_pred_svm))
print("Precision: ", precision_score(y_test, y_pred_svm, average='macro'))
print("Recall: ", recall_score(y_test, y_pred_svm, average='macro'))
print("F1 score: ", f1_score(y_test, y_pred_svm, average='macro'))


# Machine Learning (Clustering)

# a. Implement a k-means clustering algorithm to cluster the data based on the sepal length and sepal width columns. Visualize the clusters with different colors.

# In[37]:


from sklearn.cluster import KMeans

# Prepare the data
X = iris[['sepal.length', 'sepal.width']]

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
y_pred = kmeans.predict(X)

# Visualize the clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('K-Means Clustering of Iris Data based on Sepal Length and Sepal Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# b. Implement a hierarchical clustering algorithm to cluster the data based on all four columns. Visualize the clusters with different colors.

# In[38]:


from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Prepare the data
X = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]

# Perform hierarchical clustering
agg = AgglomerativeClustering(n_clusters=3)
y_pred = agg.fit_predict(X)

# Visualize the clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('Hierarchical Clustering of Iris Data based on All Four Columns')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# c. Implement a DBSCAN clustering algorithm to cluster the data based on all four columns. Visualize the clusters with different colors.

# In[39]:


from sklearn.cluster import DBSCAN

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X)

# Visualize the clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('DBSCAN Clustering of Iris Data based on All Four Columns')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# d. Implement a Gaussian Mixture Model clustering algorithm to cluster the data based on all four columns. Visualize the clusters with different colors.

# In[40]:


from sklearn.mixture import GaussianMixture

# Perform Gaussian Mixture Model clustering
gmm = GaussianMixture(n_components=3)
y_pred = gmm.fit_predict(X)

# Visualize the clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred, s=50, cmap='viridis')
plt.title('Gaussian Mixture Model Clustering of Iris Data based on All Four Columns')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()


# e. Evaluate the performance of each clustering algorithm using the silhouette score.

# In[41]:


from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


# In[42]:


# Prepare the data
X = iris[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']]

# Calculate silhouette scores for each clustering algorithm
kmeans_silhouette = silhouette_score(X, KMeans(n_clusters=3).fit_predict(X))
hierarchical_silhouette = silhouette_score(X, AgglomerativeClustering(n_clusters=3).fit_predict(X))
dbscan_silhouette = silhouette_score(X, DBSCAN(eps=0.5, min_samples=5).fit_predict(X))
gmm_silhouette = silhouette_score(X, GaussianMixture(n_components=3).fit_predict(X))

# Print silhouette scores for each algorithm
print("KMeans Silhouette Score: ", kmeans_silhouette)
print("Hierarchical Clustering Silhouette Score: ", hierarchical_silhouette)
print("DBSCAN Silhouette Score: ", dbscan_silhouette)
print("Gaussian Mixture Model Silhouette Score: ", gmm_silhouette)


# In[ ]:




