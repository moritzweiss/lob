This text file contains some documentation. better move this to confluence. 

train_batch_size:
In the context of RLlib, train_batch_size is a configuration parameter that determines the number of environment steps to be collected before an update is performed on the policy. 
It is used to control the amount of data collected for training in each iteration. You can set this parameter in the configuration dictionary when running RLlib training.


sgd_minibatch_size:
In the context of RLlib, sgd_minibatch_size is a configuration parameter that determines the size of mini-batches used during Stochastic Gradient Descent (SGD) optimization. It is the number of samples used for each update step within an iteration. The total train_batch_size is divided into smaller mini-batches of size sgd_minibatch_size for multiple update steps.


num_sgd_iter:
It represents the number of passes to make over each training batch during the optimization process. In other words, it controls how many times the algorithm iterates over the training data to update the model weights during each training step.

