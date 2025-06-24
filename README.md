# The project consists of 
debug_visualizations                   -> results of the after excuting the code
tiny-imagenet-200                      -> dataset
train                                  -> part of the dataset picked up for training small
val                                    -> part of the dataset picked up for validating small
enhanced_swin_model.py                 -> code of our model
requirements.txt                       -> required dependencies to run the project locally
tiny-imagenet-downloader.py            -> for downloading the tiny imagenet dataset
train_small.sh                         -> shell script written to automatically train the model with small subset of the dataset
traning-run-script.sh                  -> shell script written for training the model on the complete dataset
training-script.py                     -> code for training the model


# Future scope and improvements
# For future enhancement:
# 1. Add data augmentation for better generalization
# 2. Implement robustness against common image modifications (JPEG compression, noise)
# 3. Add progressive training strategy (curriculum learning)
# 4. Implement adversarial training against steganalysis detectors

 
