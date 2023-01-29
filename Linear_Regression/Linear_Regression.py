try:
    import logging
    import numpy as np

except ImportError:
    print("please install required packages for import(numpy)...")

class linear_regression:
    """
    linear regression class with fit, pred methods
    """

    def __init__(self, slope=1, intercept=1, learning_rate=0.00001, epochs=100000):

        # assert checking
        assert type(slope) == type(1.0) or type(slope) == type(1)
        assert type(intercept) == type(1.0) or type(intercept) == type(1)
        assert type(learning_rate) == type(1.0) or type(learning_rate) == type(1)
        assert type(epochs) == type(1.0) or type(epochs) == type(1)

        # initialize
        self.slope = slope
        self.intercept = intercept
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, x_train, y_train, mode=10):
        """
        read x_train, y_train and mode(DEBUg - return 5 paramertes , INFO - returns 2 parameters).
        apply linear regression, if the y_pred have huge difference from y_actal we need to update the sloper and intercept.
        updation of slope requires derivative of slope, learning_rate and slope
        updation of intercept requires derivative of intercept, learning_rate and intercept
        repeat the process with difference epochs and leaning_rates, if the expected slope and intercept we will reach

        : param: x_train,
        : param: y_train,
        : param: mode
        : return: slope, intercept, loss_history, slope_history, intercept_history
        """

        # assert checking
        assert type(x_train) == type([]) or type(x_train) == type(()) or type(np.array([]))
        assert type(y_train) == type([]) or type(y_train) == type(()) or type(np.array([]))
        assert type(mode) == type(1)

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

                    # calculate loss - MSE - Mean Squard Error
                    diff = (y_pred - y)
                    loss = diff ** 2
                    # print(f"loss: {loss}")

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

        except ValueError:
            print("please check your fit parameters and try again")

    def pred(self, x_test):

        """
        read x_test
        apply linear regression, and find y_predicted with updated new slope and intercept
        return the y_predicted

        : param: x_test,
        : return: y_predicted
        """

        # assert checking
        assert type(x_test) == type([]) or type(x_test) == type(()) or type(np.array([]))

        try:
            y_hat_pred = []

            for xt in x_test:
                y_hat = (self.slope * xt) + self.intercept

                y_hat_pred.append(y_hat)

            return y_hat_pred

        except ValueError:
            print("please your x_test value and try again")

if __name__ == '__main__':
    pass
