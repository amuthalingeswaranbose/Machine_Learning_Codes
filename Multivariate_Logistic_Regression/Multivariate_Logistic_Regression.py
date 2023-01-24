import numpy as np

class Multivariate_Logistic_Regression:
     
    def __init__(self, slope_count=1, intercept=1, learning_rate=0.001, epochs=1000):
        
        # initialize
        self.slope = np.ones(slope_count, dtype = int)
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
      
    def fit(self, x_train, y_train):
        
        loss_history = []
        slope_history = []
        intercept_history = []
        
        # train data row and column
        row, column = x_train.shape
        
        for epoch in range(self.epochs):
            
            single_epoch_losses = []
            
            for x, y in zip(x_train, y_train):                
            
                # y = (m * x) + c - linear eqation
                mult_of_slope_and_input = np.array([(self.slope[i]*x[i]) for i in range(column)])
                sum_mult_of_slope_and_input = sum(mult_of_slope_and_input)
                y_pred = sum_mult_of_slope_and_input + self.intercept
                
                # Sigmoid function
                sigmoid_of_y_pred = 1 / (1 + np.exp(-y_pred))   
                
                # calculate loss 
                loss = -(y*np.log(sigmoid_of_y_pred) + (1-y)*np.log(1-sigmoid_of_y_pred))
                
                # append single_epoch_losses loss
                single_epoch_losses.append(loss)
                
                # Find Derivatives of slope and intercept
                derivative_of_slope = np.array([(sigmoid_of_y_pred - y)*x[j] for j in range(column)])
                
                derivative_of_intercept = (sigmoid_of_y_pred - y)
                
                # Update slope and intercept
                self.slope = np.array([(self.slope[k] - (self.learning_rate * derivative_of_slope[k])) for k in range(column)])
                self.intercept -= self.learning_rate * derivative_of_intercept
            
            average_epoch_loss = sum(single_epoch_losses) / len(x_train)
            print(f"iteration - {epoch} -> loss: {average_epoch_loss}, self.slope: {self.slope}, self.intercept: {self.intercept}")  
            
            loss_history.append(average_epoch_loss)
            slope_history.append(self.slope)
            intercept_history.append(self.intercept) 
            
        return self.slope, self.intercept, loss_history, slope_history, intercept_history
       
       
    def pred(self, x_test):
        
        y_hat_pred = []
        
        # test data row and column
        row, column = x_test.shape
        
        for xt in x_test:
        
            mult_of_slope_and_test_data = np.array([(self.slope[i]*xt[i]) for i in range(column)])
            sum_mult_of_slope_and_test_data = sum(mult_of_slope_and_test_data)
            y_hat = sum_mult_of_slope_and_test_data + self.intercept
            sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))
            
            if sigmoid_of_y_hat >= 0.5:   
            
                y_hat_pred.append(1)
                
            else:
            
                y_hat_pred.append(0)
        
        return y_hat_pred
        

    def pred_porb(self, x_test):
    
        y_hat_pred_prob = []
        
        # test data row and column
        row, column = x_test.shape
        
        for xt in x_test:
                            
            mult_of_slope_and_test_data = np.array([(self.slope[i]*xt[i]) for i in range(column)])
            sum_mult_of_slope_and_test_data = sum(mult_of_slope_and_test_data)
            y_hat = sum_mult_of_slope_and_test_data + self.intercept
            
            sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))
            
            y_hat_pred_prob.append(sigmoid_of_y_hat)
        
        return y_hat_pred_prob
