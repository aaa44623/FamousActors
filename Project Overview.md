<h2>The project's goal was to explore a dataset consisting of images of famous actors, apply data science techniques to analyze the images and gain experience in handling real-world data.</h2>

<h2>Data Preprocessing:</h2>
Organized images by actor names into corresponding folders(dataset link is shared within the project).
Implemented a Python script to resize and normalize the images, ensuring they were in a consistent format suitable for analysis.
<h2>Feature Extraction:</h2>
Employed pre-trained models (like VGG16) to extract features from the images.
The extracted features were then prepared for further analysis, maintaining the association between the features and the corresponding actors.
<h2>Dimensionality Reduction and Visualization:</h2>
Applied PCA and t-SNE, two-dimensionality reduction techniques, to visualize the high-dimensional feature data in two-dimensional space.
t-SNE provided a more distinct visualization of clusters than PCA, revealing the underlying structure of the data.
<h2>Clustering:</h2>
Conducted K-Means clustering on the feature space to group the images into clusters.
The clustering results were visualized on the t-SNE plot(attached in the directory), revealing significant overlaps among clusters representing different actors.
<h2>Analysis of Overlaps:</h2>
Investigated the overlapping clusters to identify which actors' images were not being distinctly grouped and
found that images of several actors were distributed across all clusters, highlighting the complexity of the data and the challenges in clustering similar high-dimensional data.
<h2>Insights:</h2>
Determined that the extracted features may not be sufficiently discriminative, and there is high intra-class variability or low inter-class variability.
Recognized the need for more sophisticated feature extraction, possibly utilizing deep learning techniques for improved results.
<h2>Conclusion:</h2>
Although distinct clusters for each actor were not achieved, the process illuminated several aspects of data science work, including data preprocessing, feature extraction, the importance of visualization, and
the iterative nature of modeling.
<h2>Possible Next Steps:</h2>
Exploring deep learning models, such as CNNs, for feature extraction to potentially achieve better clustering results.
Refining the clustering approach, adjusting the number of clusters, or experimenting with different algorithms.
Considering other forms of data augmentation and preprocessing to enhance the model's ability to distinguish between different actors.
