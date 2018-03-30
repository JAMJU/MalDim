from kernel_creation import get_array
from vocabulary import get_label_list
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.colors as mpc
colors = mpc.cnames.keys()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mat = get_array( 'data/created/vector_input.csv')
mat_test = get_array('data/created/vector_test_input.csv')
lab = get_label_list('data/label.csv')
color_lab = [colors[l] for l in lab]

"""pca = PCA(n_components=1000)
mat_new = pca.fit_transform(mat, lab)
comp = pca.components_
with open("data/created/components.csv", 'w') as f:
    for i in range(len(comp)):
        for j in range(comp.shape[1]):
            f.write(str(comp[i,j]) + ",")
        f.write('\n')
print mat_new.shape

mat_test_pca = pca.transform(mat_test)
print mat_test_pca.shape"""

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(300):
    ax.scatter(mat_new[:
               i,0], mat_new[i,1], mat_new[i,2], c=color_lab[i])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()"""

"""for i in range(2000):
    plt.scatter( mat_new[i,0], mat_new[i,1], c=color_lab[i])
plt.show()"""

