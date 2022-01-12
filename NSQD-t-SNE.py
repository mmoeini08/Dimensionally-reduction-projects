import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=150) #to control what is printed: 'suppress=True' prevents exponential prints of numbers, 'precision=5' allows a max of 5 decimals, 'linewidth'=150 allows 150 characters to be shown in one line (thus not cutting matrices)
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.preprocessing import LabelEncoder #To switch categorical letters to numbers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold.t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
file_name = 'TSS'
data = pd.read_csv(file_name + '.csv', header=0, index_col=0)
print("{0} rows and {1} columns".format(len(data.index), len(data.columns)))
print("")
X1 = data.DA
X2 = data.PR1
X3 = data.PI1
X4 = data.PC
X5 = data.PI
X6 = data.PO
X7 = data.PF
X8 = data.IP
X9 = data.AD
X10 = data.PR
X11 = data.RO
X = np.column_stack((X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11))
print("Covariance matrix")
print(np.cov(X, rowvar=0).round(2)) #rowvar=0 means that each column is a variable. Anything else suggest each row is a variable.
print('')
print("Here 1") #print to know where you are or to check if a bug exists
a = np.linalg.eigvals(np.cov(X, rowvar=0))
print(a/a.sum()) #To show that percentage variance explained by components is the eigenvalues
print('')
print("Here 2")
print('')
print("Correlation Coefficients")
print(np.corrcoef(X, rowvar=0).round(2))
print("")
ncompo = int(input("Number of components to study:"))
print("")
tsne = TSNE(n_components=ncompo, init='random', random_state=0)
Y=tsne.fit(X)
X_new = TSNE(n_components=2).fit_transform(X,Y)
print(X_new)
before_after = input("Before / After Plot (Y):")
if before_after == 'Y' or before_after == 'y':
    le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
    le.fit(data.TSS)
    number = le.transform(data.TSS)
    colormap = np.array(['blue', 'green', 'orange', 'red'])
    fig = plt.figure(figsize=(12, 110))
    ax = fig.add_subplot(121) #Figure to have 1 row, 2 plots, focusing now on first plot
    ax.scatter(X1, X2, c=colormap[number])
    ax.set_xlabel('DA')
    ax.set_ylabel('RO')    
    ax = fig.add_subplot(122)
    ax.scatter(X_new[:,0], X_new[:,1], c=colormap[number])
    ax.set_xlabel('TS1')
    ax.set_ylabel('TS2')
    plt.savefig(file_name + '_before_after', dpi=300)
    plt.show()



##Fun 3D Plot
# plot_3D = input("3D Plot (Y):")
# if plot_3D == 'Y' or plot_3D == 'y':
#     le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
#     le.fit(data.Letter)
#     number = le.transform(data.Letter)
#     colormap = np.array(['blue', 'green', 'orange', 'red'])
    
#     fig = plt.figure(figsize=(12, 4))
#     ax = fig.add_subplot(121, projection='3d')
#     ax.scatter(X4, X5, X6, c=colormap[number])
#     ax.set_xlabel('MT1')
#     ax.set_ylabel('MT2')
#     ax.set_zlabel('Final')
    
#     ax = fig.add_subplot(122, projection='3d')
#     ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2], c=colormap[number])
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_zlabel('PC3')
    
#     plt.savefig(file_name + '_3D', dpi=300)
#     plt.show()

































#Find the PCA
# tsnefit = tsne.fit(X) #Use all data points since we are trying to figure out which variables are relevant

# print("Mean")
# print(tsnefit.mean_)
# print("")
# print("t-distributed stochastic neighboring embedding results")
# print(tsnefit.components_)
# print("")
# print("Percentage variance explained by components")
# print(tsnefit.explained_variance_ratio_)
# print("")

#print(X)
# X_new = tsne.transform(X)
#print(X_new)


#Plot percentage variance explained by components 
# perc = tsnefit.explained_variance_ratio_
# perc_x = range(1, len(perc)+1)
# plt.plot(perc_x, perc, "r--", marker=".", markersize=20)
# plt.xlabel('Components')
# plt.ylabel('Percentage of Variance Explained')
# plt.title('PCA Anlaysis')
# plt.savefig(file_name + '_pervar', dpi=300)
# plt.show()



#Before and After
#Use AvgHw and AvgQuiz so that X_new[:,0] and X_new[:,1] are always AvgHW and AvgQuiz
# before_after = input("Before / After Plot (Y):")
# if before_after == 'Y' or before_after == 'y':
#     le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
#     le.fit(data.Letter)
#     number = le.transform(data.Letter)
#     colormap = np.array(['blue', 'green', 'orange', 'red'])
    
#     #Create empty figure
#     fig = plt.figure(figsize=(12, 4))
    
#     ax = fig.add_subplot(121) #Figure to have 1 row, 2 plots, focusing now on first plot
#     ax.scatter(X1, X2, c=colormap[number])
#     ax.set_xlabel('AvgHW')
#     ax.set_ylabel('AvgQuiz')
    
#     ax = fig.add_subplot(122)
#     ax.scatter(X_new[:,0], X_new[:,1], c=colormap[number])
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
    
#     plt.savefig(file_name + '_before_after', dpi=300)
#     plt.show()



# #Fun 3D Plot
# plot_3D = input("3D Plot (Y):")
# if plot_3D == 'Y' or plot_3D == 'y':
#     le = LabelEncoder() #used to turn categorical letters to numbers: 0, 1, 2, 3
#     le.fit(data.Letter)
#     number = le.transform(data.Letter)
#     colormap = np.array(['blue', 'green', 'orange', 'red'])
    
#     fig = plt.figure(figsize=(12, 4))
#     ax = fig.add_subplot(121, projection='3d')
#     ax.scatter(X4, X5, X6, c=colormap[number])
#     ax.set_xlabel('MT1')
#     ax.set_ylabel('MT2')
#     ax.set_zlabel('Final')
    
#     ax = fig.add_subplot(122, projection='3d')
#     ax.scatter(X_new[:,0], X_new[:,1], X_new[:,2], c=colormap[number])
#     ax.set_xlabel('PC1')
#     ax.set_ylabel('PC2')
#     ax.set_zlabel('PC3')
    
#     plt.savefig(file_name + '_3D', dpi=300)
#     plt.show()
