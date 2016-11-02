# Text2Vis

Text2Vis is a family of neural network models aimed at learning a mapping from short textual descriptions to visual features, so that one can search for images by simply providing a short descriptions of it.

Text2Vis includes: (i) a sparse version, that takes a one-hot vector represeting the textual description as input; (ii) a dense version where words are embedded and given as input to an LSTM conditioning the last memory state to the visual space; and (iii) a Wide & Deep model, that combines both sparse and dense representations.

Note: to train the model, you need the visual features associated to the MsCOCO image repository. In our experiments, we considered the fc6 and fc7 layers of the Hybrid CNN. They were however too heavy to add them to the repository, but if you need them we will be very happy to share ours with you!



