# Domain Specific Knowledge Base for Question Answering

The goal of this project is to create a domain specific knowledge base 
that is easily queryable in order to better answer multiple choice questions.
The knowledge base is built with the Graph Neural Network (GNN) architecture<sup>1,2</sup> 
which is comprised of nodes *V*, edges *E* and a universal attribute *U* updated with the 
message passing system as seen in the image<sup>1</sup> below:

![message passing](https://github.com/kinne174/knowledge_graph/blob/master/pictures/battaglia_message.PNG)

 The models used on top of the GNN are Recurrant Neural Networks (LSTM), and 
 Multi Layer Perceptrons.
 
 The multiple choice questions are from the ARC<sup>3</sup> collection that were
 collected from Elementary-level standardized tests focusing on the topic of
 science. We restrict the focus even further by imposing domains (space, rocks, etc.)
 from which we draw the questions from. The knowledge base is built using the
 unlabeled corpus of information provided by ARC<sup>3</sup> which is a collection
 of ~14 million sentences gathered to help answer the questions. By re imagining question
 answering as sentence completion we can use a semi-supervised approach to train
 our knowledge base through masking. 
 
 One iteration of training can be seen below:
 
 ![my network](https://github.com/kinne174/knowledge_graph/blob/master/pictures/my_network.PNG)

This repository is made up of the main file run_kb.py which includes the data
loading, training loop and evaluation. The utils_*.py files are supporting files
used in run_kb.py.

### Citations
1 Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arXiv preprint arXiv:1806.01261, 2018

2 Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2008

3 https://allenai.org/data/arc