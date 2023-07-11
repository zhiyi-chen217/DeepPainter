# DeepPainter

Please first install the packages in requirements.txt
the data is accessible through https://polybox.ethz.ch/index.php/s/9oB0EGdHV1qBsIC

# The pre-trained checkpoints are available in the checkpoint folder

x_cae.pth contains the checkpoint of the convolutional autoencoder model that is trained for at most x epoch (might be less due to early stopping)

x_cae.pth contains the checkpoint of the classifier model that is trained for at most x epoch (might be less due to early stopping)

# Experiments
We randomly selected 5k images from the wikiart dataset and used these images to pretrain the convolutional autoencoder. The number of epochs is decided using the early stopping mechanism.
Then we trained the classifier using a dataset containing 3159 images from Vincent Van Gogh, Rembrandt and 
Pierre-Auguste Renoir. The dataset is split into training and testing sets with ration 9:1. The testing set is further split into two sets: one for validation, one for testing

The average accuracy we achieved on this dataset is around 85%.


