# Perceptron-Classifiers
Perceptron classifiers (vanilla and averaged) to identify hotel reviews as either true or fake, and either positive or negative.
We may using the word tokens as features, or any other features we can devise from the text.

# Data
The training and development data are available:

One file train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). The first 3 tokens in each line are:
a unique 7-character alphanumeric identifier
a label True or Fake
a label Pos or Neg
These are followed by the text of the review.
One file dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).
One file dev-key.txt with the corresponding labels for the development data, to serve as an answer key.

# Programs
Two programs: perceplearn.py will learn perceptron models (vanilla and averaged) from the training data, and percepclassify.py will use the models to classify new data. The learning program will be invoked in the following way:

> python perceplearn.py /path/to/input

The argument is a single file containing the training data; the program will learn perceptron models, and write the model parameters to two files: vanillamodel.txt for the vanilla perceptron, and averagedmodel.txt for the averaged perceptron. The format of the model files is up to you, but they should follow the following guidelines:

The model files should contain sufficient information for percepclassify.py to successfully label new data.
The model files should be human-readable, so that model parameters can be easily understood by visual inspection of the file.
The classification program will be invoked in the following way:

> python percepclassify.py /path/to/model /path/to/input

The first argument is the path to the model file (vanillamodel.txt or averagedmodel.txt), and the second argument is the path to a file containing the test data file; the program will read the parameters of a perceptron model from the model file, classify each entry in the test data, and write the results to a text file called percepoutput.txt in the same format as the answer key.

