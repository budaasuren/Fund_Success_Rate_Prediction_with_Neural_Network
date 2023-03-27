# Fund_Success_Rate_Prediction_with_Neural_Network

## Overview of the analysis:

The purpose of the analysis is to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup using machine learning and neural networks and the features in the provided dataset.

The data used in this analysis is a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years.


##  Final Results: 
Loss: 0.6487845182418823, Accuracy: 0.7309614419937134


## Data Preprocessing Work Flow: 

* Defined the IS_SUCCESSFUL variable as the target variable since the purpose of this analysis is to predict this varaible.
* The remaining variables are considered the features of the model. 
* The name and ID columns were removed as they would not have any influence in the final outcome. 
* For APPLICATION_TYPE and CLASSIFICATION features, created OTHER category for both of them to decrease the number of unique values in these features as they had too many rare values. In other words, condensed the rare values by binning them into OTHER category.
* Transformed the all the categorical features into numeric values using pandas'get_dummies method.
* Splited the preprocessed data into a training and testing dataset.
* Normalized the numeric features for both test and training datasets using Standard Scaler.


## Model Building (selecting hyperparameters) and Training Work Flow:
* Compiling: chose Adam as an optimizer, learning_rate as 1e-3, and  metrics as accuracy.
* Training: Created a callback that saves the model's weights every five epochs, trained the model at 100 epochs.
* Model Optimization: Three different approach were used to optimize the model performance.
  * Removed the outliers from ASK_AMT feature
  * Removed the STATUS feature as there was high imbalance in this feature ( 1=33608, 0=5)
  * Added one more hidden layer
* Evaluating the Model: for the training dataset, loss: 0.5139 - accuracy: 0.7506. Versus, for the testing dataset,Loss: 0.6487845182418823, Accuracy: 0.7309614419937134. The result is sligthly better on training dataset.

As you can see from the image below, the model has 3 hidden layers, 325 neorons (166, 166, 83 for each layer) because the number of features is 83, thus multiplied that number by 2 for the first two layers. Then used the logic to decreace the neoron number as the model progress to next hidden layer. 


![Screen Shot 2023-03-26 at 8 26 14 PM](https://user-images.githubusercontent.com/113545468/227817137-676a5885-1750-4641-96db-aa08fd083a6f.png)


The activation function used for hidden layers is RELU as RELU is industry common practice, for the output layer, SIGMOID activation function is used. The reason I chose sigmoid in the output layer, is because our final result is categorized into binary (0 or 10) classification.

## Summary: 
As the accuraccy is 0.73 for this model, I would not recommend the model to move to the production phase. Also, I recommend Decision tree based models especially Randon Forest to predict the outcome better as there are 83 features in the model and the dataset is structured. 

