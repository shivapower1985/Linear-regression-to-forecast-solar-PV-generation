#####Linear Regression program to predict/forecast solar PV generation #####

#####Created by: Shivashankar Sukumar#####


#####ABOUT the program: To forecast or predict the current day solar PV generation data, latest...#####
#####...21 days solar PV generation data is used#####

#####In this work 17 days data is used for training and 4 days data is used for testing#####

#####Import packages#####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#####Reading .CSV file which has latest solar PV generation data#####
data = pd.read_csv("sample19.csv",header=None)
df=pd.DataFrame(data)

#converting the Panda data frame into Numpy array
df=df.to_numpy()

#taking transpose of input data
df=np.transpose(df)

# Split the data into training/testing sets
###creating training from whole PV data sets and reshape into single coloumn vector
pv_train = df[:-4]
row_train=pv_train.shape[0]#tells no of rows
col_train=pv_train.shape[1]#tells no of cols
no_elements_train=row_train*col_train
pv_train=np.reshape(pv_train,(no_elements_train,1))


###creating testing from whole PV data sets and reshape into single coloumn vector
pv_test = df[-4:]
row_test=pv_test.shape[0]#tells no of rows
col_test=pv_test.shape[1]#tells no of cols
no_elements_test=row_test*col_test
pv_test=np.reshape(pv_test,(no_elements_test,1))


###create x_train and y_train from the training data set
x_train=pv_train[0:(len(pv_train)-1)]
y_train=pv_train[1:len(pv_train)]



###create x_test and y_test from the testing data set
x_test=pv_test[0:(len(pv_test)-1)]
y_test=pv_test[1:len(pv_test)]


# Create linear regression object
regr = linear_model.LinearRegression()


# Train the model using the training sets
regr.fit(x_train, y_train)


# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))



# Plot outputs - Actual vs predicted

time=range(0,len(y_pred)) #NOTE: This is a time series plot therefore the plot should be time vs actual and time vs predicted
                          #therefore creating time horizon in hours should be same as len(predicted or actual)

plot_1, =plt.plot(time,y_test, color='red', linewidth=2)
plot_2, =plt.plot(time,y_pred,color='blue', linewidth=2, linestyle='--')
plt.legend([plot_1, plot_2], ["PV_actual", "PV_predicted"])

plt.show()

#####END of program#####


