import logging
import numpy as np
class univariate_logistic_regression:
    """
    univariate logistic regression class with fit, pred and pred_prob methods
    """
    def __init__(self, slope: int = 1,
                 intercept: int = 1,
                 learning_rate: float = 0.001,
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
    def fit(self, x_train: [list, tuple, np.ndarray],
            y_train: [list, tuple, np.ndarray],
            mode: int):

        """fit the data using linear regression model
            Parameters:
            x_train: training input,
            y_train: training ouput,
            mode: mode of training (DEBUG of INFO),
            Returns:
            if mode is DEBUG
                Returning int: new slope, int: new intercept, list: loss history, list: slope history, list: intercept history
            else
                Returning int: new slope, int: new intercept
        """
        # assert checking
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

        if mode == logging.DEBUG:
            loss_history = []
            slope_history = []
            intercept_history = []

        for epoch in range(self.epochs):

            if mode == logging.DEBUG:
                single_epoch_losses = []

            for x, y in zip(x_train, y_train):
                # y = (m * x) + c - linear eqation
                y_pred = (self.slope * x) + self.intercept

                # Sigmoid function
                sigmoid_of_y_pred = 1 / (1 + np.exp(-y_pred))

                # calculate loss
                loss = -(y * np.log(sigmoid_of_y_pred) + (1 - y) * np.log(1 - sigmoid_of_y_pred))

                if mode == logging.DEBUG:
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

    def pred(self, x_test: [list, tuple, np.ndarray]):

        """predict the data using linear regression model with updated slope and intercept
            Parameters:
            x_test: training input,
            Returns:
            :Returning y_predicted
        """

        # assert checking
        assert ((isinstance(x_test, list) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, list) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.int) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.float)) == True

        y_hat_pred = []

        for xt in x_test:

            y_hat = (self.slope * xt) + self.intercept
            sigmoid_of_y_hat = 1 / (1 + np.exp(-y_hat))

            if sigmoid_of_y_hat >= 0.5:
                y_hat_pred.append(1)

            else:
                y_hat_pred.append(0)

        return np.array(y_hat_pred)

    def pred_prob(self, x_test: [list, tuple, np.ndarray]):

        """predict probability the data using linear regression model with updated slope and intercept
            Parameters:
            x_test: training input,
            Returns:
            Returning y_predicted-prob
        """
        # assert checking
        assert ((isinstance(x_test, list) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, list) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.int) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.float)) == True

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
