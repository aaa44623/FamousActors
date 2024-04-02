# Ahmed Al Balushi
# Popularity of Actors
# link of Dataset: https://www.kaggle.com/code/meeratif/famous-actors/notebook
# This Project aims to find the popular actor of the 8 actors, analyzing every actor images and draw a conclusion from it.

"""'' Part One, Resize the image and normalize it, the code is designed to run recursivly for main folder and actor folder -- mainfolder/adamsandlar/1.jpg 
import cv2
import numpy as np
import os

# Specify the path for the dataset
dataset_path = 'D:\RandomProj\Datasets\Most Famous Actors of All Time'

# Define the dimensions to resize images to
width = 150
height = 150

# Iterate over all actor folders in the dataset
for actor_folder in os.listdir(dataset_path):
    actor_folder_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_folder_path):
        # Create a specific folder for processed images of each actor
        output_actor_folder = os.path.join('D:\RandomProj\Datasets', actor_folder)
        if not os.path.exists(output_actor_folder):
            os.makedirs(output_actor_folder)
        
        # Iterate over all images in actor's folder
        for filename in os.listdir(actor_folder_path):
            # Construct the full image path
            img_path = os.path.join(actor_folder_path, filename)
            if os.path.isfile(img_path):
                # Read the image
                img = cv2.imread(img_path)

                if img is not None:
                    # Resize the image
                    img_resized = cv2.resize(img, (width, height))

                    # Normalize the image
                    img_normalized = img_resized / 255.0
                    
                    # Save the pre-processed image to the respective actor’s output folder
                    output_path = os.path.join(output_actor_folder, filename)
                    # Convert the normalized image to the range [0, 255] before saving
                    cv2.imwrite(output_path, img_normalized * 255)

""" """End of Part 1"""

"""''Part 2 Begins here''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the VGG16 model pre-trained on ImageNet data
model_vgg16 = VGG16(
    weights="imagenet", include_top=False
)  # include_top=False to exclude the final fully connected layers.

# Specify the path to the preprocessed dataset
dataset_path = 'E:\RandomProj\Datasets\ResizedNormalized' #its D: in laptop (The hardrdrive, E: in PC)

# Define a dictionary to store extracted features
features_dict = {}

# Iterate over all actor folders in the dataset
for actor_folder in os.listdir(dataset_path):
    actor_folder_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_folder_path):
        # Define a list to store features of each actor’s images
        actor_features = []

        # Iterate over all images in actor's folder
        for filename in os.listdir(actor_folder_path):
            # Construct the full image path
            img_path = os.path.join(actor_folder_path, filename)
            if os.path.isfile(img_path):
                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Extract features using VGG16
                features = model_vgg16.predict(img_array)
                actor_features.append(features.flatten())

        # Store the extracted features to the dictionary
        features_dict[actor_folder] = actor_features

# Optionally: Save the extracted features for later use
np.save("extracted_features.npy", features_dict)

'Part 2 Ends here""" """"""

"""'Part 3 begins here Reduction technique, USED PAC at first realized that approx 90% of clusters are overlapping, changed it to t-SNE technique for better result
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the extracted features
features_dict = np.load('extracted_features.npy', allow_pickle=True).item()

# Prepare the data for PCA
actors = list(features_dict.keys())
X = []  # Feature vectors
y = []  # Corresponding labels (actor names)

for actor, features in features_dict.items():
    X.extend(features)
    y.extend([actor] * len(features))

X = np.array(X)
y = np.array(y)

''
# Perform PCA  # one reduction technique
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 6))
for actor in actors:
    mask = y == actor
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=actor)

plt.legend()
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Actor Features')
plt.show()

''
#another reduction technique
# Perform t-SNE
X_tsne = TSNE(n_components=2).fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 6))
for actor in actors:
    mask = y == actor
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=actor)

plt.legend()
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Actor Features')
plt.show()


# Part 3 ends here
""" """
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the VGG16 model pre-trained on ImageNet data
model_vgg16 = VGG16(
    weights="imagenet", include_top=False
)  # include_top=False to exclude the final fully connected layers.

# Specify the path to the preprocessed dataset
dataset_path = "D:\RandomProj\Datasets\ResizedNormalized"  # its D: in laptop (The hardrdrive, E: in PC)

# Define a dictionary to store extracted features
features_dict = {}

# Iterate over all actor folders in the dataset
for actor_folder in os.listdir(dataset_path):
    actor_folder_path = os.path.join(dataset_path, actor_folder)
    if os.path.isdir(actor_folder_path):
        # Define a list to store features of each actor’s images
        actor_features = []

        # Iterate over all images in actor's folder
        for filename in os.listdir(actor_folder_path):
            # Construct the full image path
            img_path = os.path.join(actor_folder_path, filename)
            if os.path.isfile(img_path):
                # Load and preprocess the image
                img = image.load_img(img_path, target_size=(150, 150))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)

                # Extract features using VGG16
                features = model_vgg16.predict(img_array)
                actor_features.append(features.flatten())

        # Store the extracted features to the dictionary
        features_dict[actor_folder] = actor_features

# Optionally: Save the extracted features for later use
np.save("extracted_features.npy", features_dict)
""" ""


### Part 4 is a combination of part 3 reduction technique and part 4 clustering

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the extracted features
features_dict = np.load("extracted_features.npy", allow_pickle=True).item()

# Prepare the data for PCA
actors = list(features_dict.keys())
X = []  # Feature vectors
y = []  # Corresponding labels (actor names)

for actor, features in features_dict.items():
    X.extend(features)
    y.extend([actor] * len(features))

X = np.array(X)
y = np.array(y)

X_tsne = TSNE(n_components=2).fit_transform(X)

#### Clustering phase begins
# Decide the number of clusters (e.g., the number of actors if known)
n_clusters = len(actors)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)  # Use the original high-dimensional features

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Evaluate the clustering
silhouette_avg = silhouette_score(X, labels)
print(f"Silhouette Score: {silhouette_avg}")

actor_to_cluster_ids = {}

for actor in actors:
    actor_indices = np.where(y == actor)[
        0
    ]  # Find all instances of a particular actor in y
    corresponding_cluster_ids = labels[
        actor_indices
    ]  # Get the corresponding cluster IDs from labels
    unique_cluster_ids = np.unique(corresponding_cluster_ids)  # Find unique cluster IDs
    actor_to_cluster_ids[actor] = unique_cluster_ids

print(actor_to_cluster_ids)

# Visualize the Clustering Result with t-SNE components
"""''
plt.figure(figsize=(10, 6))
# adding legend to the plot
for actor in set(y):
    mask = y == actor
    # mask = np.array(actors) == actor  # replace with your actual variable
    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=actor)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")  # legend part ends here
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis")
plt.title("t-SNE with K-Means Clustering Labels")
plt.show()



# Assuming `labels` contain the cluster IDs assigned by K-Means
overlapping_clusters = [
    -14,
    0,
    8.02,
]  # Update this list based on your actual overlapping cluster IDs from your t-SNE or clustering visualization

# Get the indices of instances in the overlapping clusters
overlapping_indices = np.isin(labels, overlapping_clusters)

# Retrieve the corresponding actors and feature vectors
overlapping_actors = y[overlapping_indices]
overlapping_features = X[overlapping_indices]

# Print the overlapping actors to inspect
print(np.unique(overlapping_actors))
"""
