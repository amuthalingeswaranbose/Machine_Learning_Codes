# Machine_Learning_Codes
  this repo contains Linear Regression, Bi-variate Logistic Regression and Uni-variate Logistic Regression class implementaion from scratch (numpy) code, train and test notebook.
  
  ## Machine Learning Algorithms,
  1. Linear Regression.
  2. Logistic Regression
    * Unvariate Logistic Regression.
    * Bivariate Logistic Regression. 
  
# 1. Linear Regression 
  Linear Regression is supervised learning algorithm. this algorithm is mainly used for predictive analysis and modeling. we use old datas for train the algorithm and predict the future. 
  ## Linear Regression Application,
    1. Stockmarket price prediction.
    2. weather prediction
    3. ect..
    
  ## Formulas,
    1. y_hat = (m * x) + c (m - slope, c - intercept)
    2. loss calculation, use MSE(Mean Squard Error)
        * difference = (y_hat - y)
        * loss = difference**2 (square the difference).
    3. upate slope and intercept and minimize the loss
        * find derivative of m, dm =  2*(y_hat - y) * x
        * find derivative of b, db =  2*(y_hat - y) 
        * update slope, m = m - (learning_rate * dm)
        * update intercept, b = m - (learning_rate * db)
        
  # Operation.
    1. use linear_regression_class_test.ipynb notebook for understand Linear Regression.
    
    
    
  # 2. Logistic Regression 
  Linear Regression is supervised learning algorithm. this algorithm is mainly used for classification problems. if we have a large dataset we can use this logistic regression for categorize. 
  ## Logistic Regression Application,
    1. cancer type classification.
    2. patient will have heart attack or not.
    3. ect..
    
  ## Types of Logistic Regression
     * Univariate Logistic Regression
     * Bivariate Logistic Regression
  
    ## Univariate Logistic Regression
      Univariate Logistic Regression use a one input data for fit a data. 
  
      ## Formulas,
        1. y_hat = (m * x) + c (m - slope, c - intercept) # we use linear eqation
        2. sigmoid of y_hat = 1 / (1 + np.exp(-y_pred))  
            * loss = -(ylog(sigmoid_of_y_pred) + (1-y)log(1-sigmoid_of_y_pred))
        3. upate slope and intercept and minimize the loss
            * find derivative of m, dm =  ((sigmoid_of_y_pred - y)* x)
            * find derivative of b, db =  (sigmoid_of_y_pred - y)
            * update slope, m = m - (learning_rate * dm)
            * update intercept, b = b - (learning_rate * db)

      # Operation.
        1. use univariate-class-test.ipynb notebook for understand Linear Regression.

 
    ## Bivariate Logistic Regression
      Univariate Logistic Regression use a more than one input data for fit a data. 
  
      ## Formulas,
        1. y_hat = (m1 * x1) + (m2 * x2) + c (m1 - slope1, m2 - slope2, c - intercept) # we use linear eqation
        2. sigmoid of y_hat = 1 / (1 + exp(-y_pred))  
            * loss = -(ylog(sigmoid_of_y_pred) + (1-y)log(1-sigmoid_of_y_pred))
        3. upate slope and intercept and minimize the loss
            * find derivative of m1, dm1 =  ((sigmoid_of_y_pred - y)* x1)
            * find derivative of m2, dm2 =  ((sigmoid_of_y_pred - y)* x2)
            * find derivative of b, db =  (sigmoid_of_y_pred - y)
            * update slope, m1 = m1 - (learning_rate * dm1)
            * update slope, m2 = m2 - (learning_rate * dm2)
            * update intercept, b = b - (learning_rate * db)

      # Operation.
        1. use bivariate-class.ipynb notebook for understand Linear Regression.
 
    
      
