# Tex2Vis

Tex2Vis is a neural net model aimed at learning a mapping from textual descriptions to visual features, so that one can search for images by simply providing a short descriptions of it.

This is a prototype implementation in Tensorflow, that can be run on Notebook.

Note: to train the model, you need the visual features associated to the MsCOCO image repository. In our experiments, we considered the fc6 layer of the Hybrid CNN. They were however too heavy to add them to the repository, but if you need them we will be very happy to share ours with you!

