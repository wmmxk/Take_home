###Summary
This repo is for a regression project, in which a continuous variable is predicted based on six numerical features and two categorical features.

Two types of model, Gaussian Process and Feedforward Neural Network, were used to make predictions (./code/FFN.ipynb, ./code/Gaussian Process.ipynb). The two models both achieved a PCC of 0.97 between prediction and truth on the training set in 5-fold-cross-validation. Finally, the average of the predictions by the two model was saved in ./final_submission/submission_ensemble.csv)


###Dependencies

keras: '2.1.3'
sklearn: '0.19.0'
pandas: '0.20.1'
numpy:'1.12.1'
scipy: '0.19.1'

