import logging
import numpy as np
class univariate_logistic_regression:
    """
    univariate logistic regression class with fit, pred and pred_prob methods
    """
    def __init__(self, slope: int = 1,
                 intercept: int = 1,
                 learning_rate: float = 0.00001,
                 epochs: int = 100000):

        # assert checking
        assert type(slope) in [int, float]
        assert type(intercept) in [int, float]
        assert type(learning_rate) in [int, float]
        assert type(epochs) in [int, float]

        # initialize
        self.slope = slope
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs
    def fit(self, x_train: np.ndarray,
            y_train: np.ndarray,
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
        assert type(x_train) in [list[float], tuple[float], np.ndarray]
        assert type(y_train) in [list[float], tuple[float], np.ndarray]
        assert type(mode) in [int, float]
        try:

            if mode == logging.DEBUG:
                loss_history = []
                slope_history = []
                intercept_history = []

            for epoch in range(self.epochs):

                single_epoch_losses = []

                for x, y in zip(x_train, y_train):
                    # y = (m * x) + c - linear eqation
                    y_pred = (self.slope * x) + self.intercept
                    
                    # Sigmoid function
                    sigmoid_of_y_pred = 1 / (1 + np.exp(-y_pred))   
                    
                    # calculate loss 
                    loss = -(y*np.log(sigmoid_of_y_pred) + (1-y)*np.log(1-sigmoid_of_y_pred))
                    
                    # append single_epoch_losses loss
                    single_epoch_losses.append(loss)
                    
                    # Find Derivatives of slope and intercept
                    derivative_of_slope = ((sigmoid_of_y_pred - y) * x)
                    derivative_of_intercept = (sigmoid_of_y_pred - y)
                    
                    # Update slope and intercept
                    self.slope -= self.learning_rate * derivative_of_slope
                    self.intercept -= self.learning_rate * derivative_of_intercept

                if mode == logging.DEBUG:

                    average_epoch_loss = sum(single_epoch_losses) / len(x_train)
                    loss_history.append(average_epoch_loss)
                    slope_history.append(self.slope)
                    intercept_history.append(self.intercept)

            if mode == logging.DEBUG:
                return self.slope, self.intercept, loss_history, slope_history, intercept_history
            else:
                return self.slope, self.intercept

        except ValueError:
            print("please check your fit parameters and try again")

    def pred(self, x_test: np.ndarray):

        """predict the data using linear regression model with updated slope and intercept
            Parameters:
            x_test (list, tuple): training input,
            Returns:
            [list,tuple]:Returning y_predicted
        """

        # assert checking
        assert type(x_test) in [list[float], tuple[float], np.ndarray]

        try:
            y_hat_pred = []

            for xt in x_test:

                y_hat = (self.slope * xt) + self.intercept
                sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))

                if sigmoid_of_y_hat >= 0.5:
                    y_hat_pred.append(1)

                else:
                    y_hat_pred.append(0)

            return np.array(y_hat_pred)

        except ValueError:
            print("please your x_test value and try again")
    def pred_prob(self, x_test: np.ndarray):

        """predict probability the data using linear regression model with updated slope and intercept
            Parameters:
            x_test (list, tuple): training input,
            Returns:
            [list,tuple]:Returning y_predicted-prob
        """
        # assert checking
        assert type(x_test) in [list[float], tuple[float], np.ndarray]

        try:
            y_hat_pred_prob = []

            for xt in x_test:

                y_hat = (self.slope * xt) + self.intercept
                sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))
                y_hat_pred_prob.append(sigmoid_of_y_hat)

            return np.array(y_hat_pred_prob)

        except ValueError:
            print("please your x_test value and try again")

if __name__ == '__main__':
    pass
