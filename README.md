# Deep-learning-assisted-template-attack

Implementation of the algorithm from the paper "The Best of Two Worlds: Deep Learning-assisted Template Attack" to guess the secret key of an encryption algorithm using its leakage traces.

To train the triplet model, run SCA_triplet_model.ipynb notebook
To perform template attack using the triplet model, run template_attack notebook in the folder ascad_template_attack

To replace template attack by deep learning template attack, run SCA_Triplet_Model_Deep_learning_Template_Attack.ipynb. It needs an already trained triplet model

To train a simple MLP, and get its performance using rank metric, run SimpleMLP.ipynb
