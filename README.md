# Machine_Learning_Codes
  this repo contains Linear Regression, Bi-variate Logistic Regression and Uni-variate Logistic Regression class implementaion from scratch (numpy) code, train and test notebook.
  
  Machine Learning Algorithms,
    1. Linear Regression.
    2. Unvariate Logistic Regression.
    3. Bivariate Logistic Regression.
  
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
    1. prepare x training samples.
    2. initialize slope and intercept as 1 
    3. predict y line
  
  
