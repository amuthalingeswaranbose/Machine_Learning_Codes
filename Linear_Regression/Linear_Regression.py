import logging
import numpy as np
class linear_regression:
    """
    linear regression class with __init__, fit and pred methods
    """
    def __init__(self, slope: int = 1,
                 intercept: int = 1,
                 learning_rate: float = 0.00001,
                 epochs: int = 100000):

        """init method
            Parameters:
            slope (int): training input,
            intercept (int): training ouput,
            learning_rate (int): training ouput,
            epochs (int): mode of training (DEBUG of INFO),
        """

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

    #def fit(self, x_train: np.ndarray, y_train: np.ndarray, mode: int):
    def fit(self, x_train: [list, tuple, np.ndarray], y_train: [list, tuple, np.ndarray], mode: int):

        """fit the data using linear regression model
            Parameters:
            x_train (list[int], tuple[int], list[int], tuple[int], np.ndarray): training input,
            y_train (list[int], tuple[int], list[int], tuple[int], np.ndarray): training ouput,
            mode (int, float): mode of training (DEBUG or others),
            Returns:
            if mode is DEBUG
                Returning int: new slope, int: new intercept, list: loss history, list: slope history, list: intercept history
            else
                Returning int: new slope, int: new intercept
        """

        # assert checking
        #assert type(x_train) in [list[int], tuple[int], list[float], tuple[float], np.ndarray]
        #assert type(y_train) in [list[int], tuple[int], list[float], tuple[float], np.ndarray]

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

                # calculate loss - MSE - Mean Squard Error
                diff = (y_pred - y)
                loss = diff ** 2
                # print(f"loss: {loss}")

                if mode == logging.DEBUG:
                    # append single_epoch_losses loss
                    single_epoch_losses.append(loss)

                # Find Derivatives of slope and intercept
                derivative_of_slope = 2 * (y_pred - y) * x
                derivative_of_intercept = 2 * (y_pred - y)

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

    def pred(self, x_test: np.ndarray):

        """predict the data using linear regression model with updated slope and intercept
            Parameters:
            x_test (list, tuple): training input,
            Returns:
            [list,tuple]:Returning y_predicted
        """
        # assert checking
        #assert type(x_test) in [list[int], tuple[int], list[float], tuple[float], np.ndarray]

        assert ((isinstance(x_test, list) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, list) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, int) for item in x_test)) or
                (isinstance(x_test, tuple) and all(isinstance(item, float) for item in x_test)) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.int) or
                (isinstance(x_test, np.ndarray) and x_test.dtype == np.float)) == True

        y_hat_pred = []

        for xt in x_test:
            y_hat = (self.slope * xt) + self.intercept
            y_hat_pred.append(y_hat)

        return y_hat_pred

if __name__=='__main__':
    pass
