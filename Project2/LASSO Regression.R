install.packages("glmnet")
install.packages('caret')
install.packages('vip')
library(glmnet) # Library containing Lasso Regression operator
library(caret) # Library to split the Data to Test and Train sets
#library(fastDummies)
library(vip)

#Importing the Data 
data=read.csv('without na.csv')

# Removing NaN values
#data=na.omit(data)
str(data)

# Converting the Logical values containing columns to integer columns
data$credit.policy=as.factor(data$credit.policy)
data$purpose = as.factor(data$purpose)
data$not.fully.paid=as.factor(data$not.fully.paid)
data$has.pub.rec=as.factor(data$has.pub.rec)
data$has.revol.bal=as.factor(data$has.revol.bal)
data$inq.last.6mths=as.integer(data$inq.last.6mths)
data$has.delinq.2yrs=as.factor(data$has.delinq.2yrs)
data$delinq.2yrs=as.integer(data$delinq.2yrs)

# glmnet library can't handle factor variable so the variables have to be converted to dummy variables
data$credit.policy.T <- ifelse(data$credit.policy == TRUE, 1, 0)
data$credit.policy.F <- ifelse(data$credit.policy == FALSE, 1, 0)
data$has.delinq.2yrs.T <- ifelse(data$has.delinq.2yrs == TRUE, 1, 0)
data$has.delinq.2yrs.F <- ifelse(data$has.delinq.2yrs == FALSE, 1, 0)
data$has.revol.bal.T <- ifelse(data$has.revol.bal == TRUE, 1, 0)
data$has.revol.bal.F <- ifelse(data$has.revol.bal == FALSE, 1, 0)
data$has.pub.rec.T <- ifelse(data$has.pub.rec == TRUE, 1, 0)
data$has.pub.rec.F <- ifelse(data$has.pub.rec == FALSE, 1, 0)
data$has.revol.bal.T <- ifelse(data$has.revol.bal == TRUE, 1, 0)
data$has.revol.bal.F <- ifelse(data$has.revol.bal == FALSE, 1, 0)
data$not.fully.paid.T<- ifelse(data$not.fully.paid == TRUE, 1, 0)
data$not.fully.paid.F<- ifelse(data$not.fully.paid == FALSE, 1, 0)




#setting the seed
set.seed(1000)

# Extracting the values of Y
Y=data$int.rate 

# Extracting the values of X
X=subset(data,select=-c(purpose,int.rate,credit.policy,not.fully.paid,has.delinq.2yrs,has.pub.rec,has.revol.bal,log.revol.bal,X,revol.util))
#X=subset(data,select=-c(purpose,int.rate,credit.policy,not.fully.paid,has.delinq.2yrs,has.pub.rec,has.revol.bal,log.revol.bal,X))
# Partition and create the index matrix of selected values
index=  createDataPartition(data$int.rate, p=0.7, list=FALSE, times=1)


# Creating Test and Training Data in the ratio of 70% and 30%
dat_train=as.matrix(X[index,])
dat_test=as.matrix(X[-index,])
#train_datx=(X[index,])
train_daty=as.matrix(Y[index])
#test_datx=as.data.frame(X[-index,])
test_daty=as.matrix(Y[-index])

# Fitting Lasso 
alpha1.fit=cv.glmnet(dat_train,train_daty,alpha=1,type.measure = 'mse',family='gaussian')

# Best Lambda
best_lambda <- alpha1.fit$lambda.min
best_lambda

# Co-efficients
coef(alpha1.fit)

# Plot
plot(alpha1.fit)

# Using the model to predict the interest rate using the test data
y_predicted <- predict(alpha1.fit, s = best_lambda, newx = dat_test)


# Calculating the total sum of squares(sst)
sst <- sum((test_daty - mean(test_daty))^2)
# Calculating the sum of squared errors (sse)
sse <- sum((y_predicted - test_daty)^2)

# Calculating the mean squared errors
mse=mean((y_predicted - test_daty)^2)
mse


# R squared
rsq=1-(sse/sst)
rsq


# Plotting the variable importance graph
(vip(alpha1.fit))
