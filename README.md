### Summary
This repo is for a regression project, in which a continuous variable was predicted based on six numerical features and two categorical features. <br />
Two types of model, Gaussian Process and Feedforward Neural Network, were used to make predictions (code for the model: ./code/FFN.ipynb, ./code/Gaussian Process.ipynb). The two models both achieved a PCC of 0.97 between prediction and truth on the training set in 5-fold-cross-validation.

Finally, the average of the predictions by the two model was saved in ./final_submission/submission_ensemble.csv)


### Dependencies
python: 3.6
keras: '2.1.3' <br />
sklearn: '0.19.0' <br />
pandas: '0.20.1' <br />
numpy:'1.12.1' <br /> 
scipy: '0.19.1' <br />

