Intelligent Q-A system
We developed a backend API service based on the Python Flask framework, using a large language model for named entity recognition, deep learning models for intent recognition, and knowledge graphs as the backend. Finally, use a large language model to summarize the output.

CMEDQA is a folder for named entity recognition dataset processing and large language model named entity recognition recognition process
Firstly, based on the corpus characteristics of public datasets, design a dataset format that conforms to the input of the big language model, convert it into a format that corresponds to the format, and further filter it according to the corpus characteristics needed, such as deleting overly long or too short sentences. After obtaining a dataset that meets the requirements, test and evaluate the performance of different big language models in the test. These evaluation indicators include accuracy A, precision P, recall R, and F1 score.

YITU is the folder for training and scoring intent recognition models
Train on the model framework using the created dataset or public dataset, and adjust parameters such as the number of labels, model structure layers, and input/output positions in config.

Templates are web pages created based on layuamimi
These web pages include the main page of the Q&A system and the Q&A history page, mainly serving as front-end display functions
QA is the process of concatenating large language models for named entity recognition, intent recognition models, and presenting backend program folders based on the Python Flask framework.

After the user inputs text, the intent recognition model recognizes the user's intent and matches it with the corresponding answer template. Then, the large language model names entities to recognize keywords, finds the corresponding information in the knowledge graph database, and fills the template slots to achieve the answer effect. The implementation of multi-modal knowledge graph is based on locally saved images. When the user needs to search for the desired node information, the backend matches the corresponding image, and then launches it to the frontend through API for visual display
