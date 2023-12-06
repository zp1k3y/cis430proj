import pandas as pd
import numpy as np
import random as rd
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt

dataset = pd.read_csv('earthquake data.csv')

# Preprocessing

# Drop qualitative values from the dataset before PCA
dataset = dataset.drop(['Date & Time', 'Lands', 'Country'], axis=1)
# The earthquake dataset is large enough that it crashes some of our Python
# environments so we've also had a backup dataset to test on, that's what this is
#dataset = pd.get_dummies(dataset, columns=['sex'])

dataset = dataset.dropna()

scaled_data = preprocessing.scale(dataset)

# The PCA algorithm

pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]

pca_df = pd.DataFrame(pca_data, columns=labels)

# Determine which PCs fall within our threshold of 0.95 and drop the rest of them
sum_var = 0
i = 0
while (sum_var < 95):
    sum_var += per_var[i]
    i += 1
pca_df.drop(['PC' + str(x) for x in range(i, len(per_var) + 1)], axis=1)

# Concatenate pca_df and scaled_data in order to calculate the correlation matrix
scaled_df = pd.DataFrame(scaled_data, columns=dataset.columns)
merge_df = pd.concat([pca_df, scaled_df], axis=1)

# Calculate the correlation matrix
correlation = merge_df.corr()

# Visualization

print(pca_df)
print(scaled_data)

# Bar chart of Percentage of Variance per Principal Component
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

print(correlation)

# Zombie code for creating a scatterplot of PCA data, it takes ages to plot 53,000
# points and isn't really that necessary so we're not spending the time to plot it

#plt.scatter(pca_df.PC1, pca_df.PC2)
#plt.title('PCA Graph')
#plt.xlabel('PC1 - {0}%'.format(per_var[0]))
#plt.ylabel('PC2 - {0}%'.format(per_var[1]))

#for sample in pca_df.index:
#    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

#plt.show()
