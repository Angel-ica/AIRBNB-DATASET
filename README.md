
# AIRBNB-DATASET
A framework to train, tune and evaluate machine learning models on several tasks that are tackled by the Airbnb team. 


The main task of the project is to build machine learning models for predicting the price, category and number of bedrooms of properties listed in the Airbnb dataset. 

Airbnb-property-listings
-----------------------------------------------------

Data cleaning i.e dropping empty rows, columns, filling null values, resizing images etc.

-----------------------------------------------------

Linear/ Logistic Regression & Classification
--------------------------------------------------
The cleaned data is now used to train regression and classification models to predict price and category respectively.
After prediction, we evaluate the loss and try to imporove our model by optimising our hyperparameters. 
The best model, hyperparameters and the metrics are stored for future reference.

------------------------------------------------------

PyTorch 
----------------------------------------------------------

PyTorch is an open source ml library based on the Torch library. It provides a variety of tools and utilities for building and training neural networks, including automatic differentiation, tensor computation with GPU acceleration, and a high-level API for building neural network models.
In this project, it is used to build neural networks to predict the price of houses and the number of bedrooms. 

The loss is evaluated and the hyperparameters are optimised to reduce the loss.

![IMG_0770](https://user-images.githubusercontent.com/87237671/222360709-18eaf730-d425-4d47-a749-b35c4615d4cf.jpg)


The best model, hyperparameters and the metrics are stored for future reference.

----------------------------------------------------------


