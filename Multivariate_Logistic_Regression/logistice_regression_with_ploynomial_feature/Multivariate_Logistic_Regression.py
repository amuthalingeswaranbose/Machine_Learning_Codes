import logging
import numpy as np


class multivariate_logistic_regression:
    """
    multivariate logistic regression class with fit, pred and pred_prob methods
    """

    def __init__(self, slope_count: int = 1,
                 intercept: int = 1,
                 learning_rate: float = 0.00001,
                 epochs: int = 100000):

        # assert checking
        assert type(slope_count) in [int, float]
        assert type(intercept) in [int, float]
        assert type(learning_rate) in [int, float]
        assert type(epochs) in [int, float]

        # initialize
        self.slope = np.ones(slope_count, dtype=int)
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, x_train: [list, tuple, np.ndarray],
            y_train: [list, tuple, np.ndarray],
            mode: int):

        """fit the data using linear regression model
            Parameters:
            x_train (list, tuple): training input,
            y_train (list, tuple): training ouput,
            mode (int, float): mode of training (DEBUG of INFO),
            Returns:
            if mode is DEBUG
                Returning int: new slope, int: new intercept, list: loss history, list: slope history, list: intercept history
            else
                Returning int: new slope, int: new intercept
        """
        # assert checking
        #assert type(x_train) in [list[float], tuple[float], np.ndarray]
        #assert type(y_train) in [list[float], tuple[float], np.ndarray]

        assert ((isinstance(x_train, list) and all(isinstance(item, int) for item in x_train)) or
                (isinstance(x_train, list) and all(isinstance(item, float) for item in x_train)) or
                (isinstance(x_train, tuple) and all(isinstance(item, int) for item in x_train)) or
                (isinstance(x_train, tuple) and all(isinstance(item, float) for item in x_train)) or
                (isinstance(x_train, np.ndarray) and x_train.dtype == np.int) or
                (isinstance(x_train, np.ndarray) and x_train.dtype == np.float)) == True

        assert ((isinstance(y_train, list) and all(isinstance(item, int) for item in y_train)) or
                (isinstance(y_train, list) and all(isinstance(item, float) for item in y_train)) or
                (isinstance(y_train, tuple) and all(isinstance(item, int) for item in y_train)) or
                (isinstance(y_train, tuple) and all(isinstance(item, float) for item in y_train)) or
                (isinstance(y_train, np.ndarray) and y_train.dtype == np.int) or
                (isinstance(y_train, np.ndarray) and y_train.dtype == np.float)) == True

        assert type(mode) in [int, float]
        try:

            if mode == logging.DEBUG:
                loss_history = []
                slope_history = []
                intercept_history = []

            # train data row and column
            #row, column = x_train.shape

            for epoch in range(self.epochs):

                if mode == logging.DEBUG:
                    single_epoch_losses = []

                for x, y in zip(x_train, y_train):
                    
                    #print(f"x: {x}, y: {y}")

                    mult_of_slope_and_input = np.array([(self.slope[i] * x[i]) for i in range(0, len(x))])
                    sum_mult_of_slope_and_input = sum(mult_of_slope_and_input)
                    y_pred = sum_mult_of_slope_and_input + self.intercept
                    #print(f"y_pred: {y_pred}")

                    # Sigmoid function
                    sigmoid_of_y_pred = 1 / (1 + np.exp(-y_pred))
                    #print(f"sigmoid_of_y_pred: {sigmoid_of_y_pred}")

                    # calculate loss
                    loss = -(y * np.log(sigmoid_of_y_pred) + (1 - y) * np.log(1 - sigmoid_of_y_pred))                
                    
                    #print(f"loss: {loss}")

                    if mode == logging.DEBUG:
                        # append single_epoch_losses loss
                        single_epoch_losses.append(loss)

                    # Find Derivatives of slope and intercept
                    derivative_of_slope = np.array([(sigmoid_of_y_pred - y) * x[j] for j in range(0, len(x))]) 
                    derivative_of_intercept = (sigmoid_of_y_pred - y)
                    
                    #print(f"derivative_of_slope: {derivative_of_slope}")
                    #print(f"derivative_of_intercept: {derivative_of_intercept}")

                    # Update slope and intercept
                    self.slope = np.array([(self.slope[k] - (self.learning_rate * derivative_of_slope[k])) for k in range(0, len(x))])
                    self.intercept -= self.learning_rate * derivative_of_intercept
                    
                    #print(f"self.slope: {self.slope}")
                    #print(f"self.intercept: {self.intercept}")

                if mode == logging.DEBUG:
                    average_epoch_loss = sum(single_epoch_losses) / len(x_train)
                    print(f"iteration - {epoch} -> loss: {average_epoch_loss}")  
            
                    loss_history.append(average_epoch_loss)
                    slope_history.append(self.slope)
                    intercept_history.append(self.intercept)

            if mode == logging.DEBUG:
                return self.slope, self.intercept, loss_history, slope_history, intercept_history
            else:
                return self.slope, self.intercept

        except ValueError:
            print("please check your fit parameters and try again")

    def pred(self, x_test: [list, tuple, np.ndarray]):

        """predict the data using linear regression model with updated slope and intercept
            Parameters:
            x_test (list, tuple): training input,
            Returns:
            [list,tuple]:Returning y_predicted
        """

        # assert checking
        #assert type(x_test) in [list[float], tuple[float], np.ndarray]

        assert ((isinstance(x_test, list) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, list) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.int) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.float)) == True

        try:
            y_hat_pred = []

            # test data row and column
            #row, column = x_test

            for xt in x_test:

                mult_of_slope_and_test_data = np.array([(self.slope[i] * xt[i]) for i in range(0, len(xt))])
                sum_mult_of_slope_and_test_data = sum(mult_of_slope_and_test_data)
                y_hat = sum_mult_of_slope_and_test_data + self.intercept

                sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))

                if sigmoid_of_y_hat >= 0.5:
                    y_hat_pred.append(1)

                else:
                    y_hat_pred.append(0)

            return np.array(y_hat_pred)

        except ValueError:
            print("please your x_test value and try again")

    def pred_prob(self, x_test: [list, tuple, np.ndarray]):

        """predict probability the data using linear regression model with updated slope and intercept
            Parameters:
            x_test (list, tuple): training input,
            Returns:
            [list,tuple]:Returning y_predicted-prob
        """
        # assert checking
        #assert type(x_test) in [list[float], tuple[float], np.ndarray]

        assert ((isinstance(x_test, list) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, list) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.int) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.float)) == True

        try:
            y_hat_pred_prob = []

            # test data row and column
            #row, column = x_test

            for xt in x_test:

                mult_of_slope_and_test_data = np.array([(self.slope[i] * xt[i]) for i in range(0, len(x))])
                sum_mult_of_slope_and_test_data = sum(mult_of_slope_and_test_data)
                y_hat = sum_mult_of_slope_and_test_data + self.intercept

                sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))

                y_hat_pred_prob.append(sigmoid_of_y_hat)

            return np.array(y_hat_pred_prob)

        except ValueError:
            print("please your x_test value and try again")


if __name__ == '__main__':
    pass
