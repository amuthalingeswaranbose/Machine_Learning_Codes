import unittest
import logging
import numpy as np
import univariate_logistic_regression as univariate_logistic_regression
class test(unittest.TestCase):
    """
    the basic class inherits .TestCase
    """

    # mode
    mode = logging.DEBUG

    # creating linear_regression object
    univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression()

    # initialize
    x_train = np.array([42.54174696, 16.5657875 , 27.4220827 , 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089  , 22.65059986,
                       13.00476328, 10.2531771 , 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619 , 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709 , 86.33721108,
                       82.00979265, 75.28765789, 85.1550368 , 74.86287242, 71.82513828,
                       71.21482288, 85.2373968 , 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872 , 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

    y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    y_pred_prob = np.array([0.997780576330649,
                             0.9999999161054437,
                             0.9999940781105815,
                             0.9974794953309152,
                             0.9999961317178886,
                             0.9999236292369255,
                             0.9999815388311645,
                             0.999999474713979,
                             0.9999989341810036,
                             0.9895860190732244,
                             0.9962045925299514,
                             0.9999928269097481,
                             0.9999997153045388,
                             0.9872666692531995,
                             0.9999990881528921,
                             0.9999999792354015,
                             0.9999999929408361,
                             0.9999955135024197,
                             0.9997862657779015,
                             0.9752050992027054,
                             0.9999999922715394,
                             0.9979035300417843,
                             0.999998492563202,
                             0.9999439071861672,
                             0.9998906523510884,
                             0.9997325758985144,
                             0.9999970671042243,
                             0.9686426708128384,
                             0.9999965546073983,
                             0.9999999702486955,
                             0.9612081271097421,
                             0.999952853113275,
                             0.9999972965473111,
                             0.9991835531224998,
                             0.9999764419026308,
                             0.9999951002375335,
                             0.9999999923565824,
                             0.99999700476169,
                             0.9999995652394368,
                             0.978989945351252,
                             0.9999998232580357,
                             0.9999999743712931,
                             0.986735915110199,
                             0.9999999773918208,
                             0.9986711240022761,
                             0.99975577276565,
                             0.9999587684399831,
                             0.9997459279897368,
                             0.9999797719731653,
                             0.9999996075208722,
                             1.901816571694748e-05,
                             0.0007170743535491236,
                             8.034991034177435e-05,
                             9.941575443322043e-05,
                             6.0267496227573414e-05,
                             8.964742322413734e-05,
                             0.00010923026962521404,
                             5.639809150709483e-06,
                             0.0014701540860665165,
                             0.00010417296688632779,
                             0.00020402982605309108,
                             3.892759737046686e-06,
                             7.792042234533488e-05,
                             0.002038235781008781,
                             0.005611079473261273,
                             1.0943253362926852e-05,
                             0.00815781719655047,
                             5.207240460316847e-05,
                             0.00017261944092742656,
                             0.0008857821029591032,
                             0.008786590271500092,
                             0.00039677319732820397,
                             7.051966653296723e-06,
                             0.0005346265143100566,
                             1.5660871835955672e-05,
                             8.544878862115895e-05,
                             0.0011910891900120699,
                             2.4895768120827453e-05,
                             0.0014066552525242339,
                             0.0046141230762679454,
                             0.005854353157381848,
                             2.410464332056642e-05,
                             0.005101927680634945,
                             0.00012739013785305637,
                             0.000625361055329386,
                             8.104265155943125e-06,
                             2.6062142085743167e-05,
                             0.007604986929794906,
                             1.3833823299737615e-05,
                             0.00400217163969398,
                             9.207243456577143e-06,
                             0.007243151213300324,
                             0.0008352207929034482,
                             1.0886426512488097e-05,
                             0.0009210779053725058,
                             0.003929078661085518,
                             0.0005044793825332768,
                             6.26131204160488e-05,
                             0.0005764221147623778,
                             8.106286936301547e-05])

    # It is a test case function to check the linear_regression.fit function - 1
    def test_1_fit(self):

        print("Start fit-1 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                self.y_train,
                mode=self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-1 test\n")

    def test_2_fit(self):

        print("Start fit-2 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(0,
                self.y_train,
                mode=self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(0,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-2 test\n")

    def test_3_fit(self):

        print("Start fit-3 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(0.0, self.y_train, mode=self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(0.0, self.y_train, mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-3 test\n")

    def test_4_fit(self):

        print("Start fit-4 test\n")

        x_train = [42.54174696, 16.5657875 , 27.4220827 , 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089  , 22.65059986,
                       13.00476328, 10.2531771 , 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619 , 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709 , 86.33721108,
                       82.00979265, 75.28765789, 85.1550368 , 74.86287242, 71.82513828,
                       71.21482288, 85.2373968 , 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872 , 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-4 test\n")

    def test_5_fit(self):

        print("Start fit-5 test\n")

        x_trian = tuple([42.54174696, 16.5657875 , 27.4220827 , 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089  , 22.65059986,
                       13.00476328, 10.2531771 , 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619 , 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709 , 86.33721108,
                       82.00979265, 75.28765789, 85.1550368 , 74.86287242, 71.82513828,
                       71.21482288, 85.2373968 , 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872 , 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(x_trian,
                                                                                                                    self.y_train,
                                                                                                                    mode=self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_trian,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-5 test\n")

    def test_6_fit(self):

        print("Start fit-6 test\n")

        x_train = ['xyz', 16.5657875 , 27.4220827 , 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089  , 22.65059986,
                       13.00476328, 10.2531771 , 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619 , 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709 , 86.33721108,
                       82.00979265, 75.28765789, 85.1550368 , 74.86287242, 71.82513828,
                       71.21482288, 85.2373968 , 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872 , 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode=self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-6 test\n")

    def test_7_fit(self):

        print("Start fit-7 test\n")

        x_train = tuple(['xyz', 16.5657875 , 27.4220827 , 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089  , 22.65059986,
                       13.00476328, 10.2531771 , 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619 , 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709 , 86.33721108,
                       82.00979265, 75.28765789, 85.1550368 , 74.86287242, 71.82513828,
                       71.21482288, 85.2373968 , 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872 , 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                x_train,
                self.y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train,
                                                              self.y_train,
                                                              mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-7 test\n")

    def test_8_fit(self):

        print("Start fit-8 test\n")

        x_train = np.array(['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                         33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                         43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                         13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                         10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                         37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                         49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                         26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                         18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                         36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                         85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                         81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                         79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                         87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                         70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                         82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                         71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                         88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                         87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                         72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                x_train,
                self.y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train,
                                                                     self.y_train,
                                                                     mode=self.mode)

        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-8 test\n")

    def test_9_fit(self):

        print("Start fit-9 test\n")
        self.logger.debug("Start fit-9 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                [0],
                self.y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit([0],
                                                                     self.y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-9 test\n")

    def test_10_fit(self):

        print("Start fit-10 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                tuple([0]),
                self.y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(tuple([0]),
                                                                     self.y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-10 test\n")

    def test_11_fit(self):

        print("Start fit-11 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                self.y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     self.y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-11 test\n")

    def test_12_fit(self):

        print("Start fit-12 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                0,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     0,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-12 test\n")

    def test_13_fit(self):

        print("Start fit-13 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                0.0,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     0.0,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-13 test\n")

    def test_14_fit(self):

        print("Start fit-14 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                [0],
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     [0],
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-14 test\n")

    def test_15_fit(self):

        print("Start fit-15 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                tuple([0]),
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     tuple([0]),
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-15 test\n")

    def test_16_fit(self):

        print("Start fit-16 test\n")

        y_train = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-16 test\n")

    def test_17_fit(self):

        print("Start fit-17 test\n")

        y_train = tuple([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(
                self.x_train,
                y_train,
                mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-17 test\n")

    def test_18_fit(self):

        print("Start fit-18 test\n")

        y_train = ['xyz', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    y_train,
                                                                                                                    mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-18 test\n")


    def test_19_fit(self):
        print("Start fit-19 test\n")

        y_train = tuple(['xyz', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    y_train,
                                                                                                                    mode=self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                     y_train,
                                                                     mode=self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-19 test\n")


    def test_20_fit(self):
        print("Start fit-20 test\n")

        y_train = np.array(['xyz', 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    y_train,
                                                                                                                    mode = self.mode)

        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                        y_train,
                                                                        mode = self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-20 test\n")

    def test_21_fit(self):
        print("Start fit-21 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode = 'heelo')
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    self.y_train,
                                                                    mode = 'hello')
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-21 test\n")

    def test_22_fit(self):
        print("Start fit-22 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode = 0)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    self.y_train,
                                                                    mode = 0)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-22 test\n")

    def test_23_fit(self):
        print("Start fit-23 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode = 0.0)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    self.y_train,
                                                                    mode = 0.0)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-23 test\n")

    def test_24_fit(self):
        print("Start fit-24 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode = [0])
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    self.y_train,
                                                                    mode = [0])
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-24 test\n")

    def test_25_fit(self):
        print("Start fit-25 test\n")

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    self.y_train,
                                                                                                                    mode = tuple([0]))
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    self.y_train,
                                                                    mode = tuple([0]))
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-25 test\n")

    def test_26_fit(self):
        print("Start fit-26 test\n")

        x_train_int = [42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                     33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                     43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                     13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                     10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                     37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                     49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                     26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                     18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                     36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                     85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                     81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                     79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                     87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                     70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                     82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                     71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                     88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                     87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                     72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]
        x_train_float = [float(x) for x in x_train_int]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(x_train_float,
                                                                                                                    self.y_train,
                                                                                                                    mode = self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train_float,
                                                                    self.y_train,
                                                                    mode = self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-26 test\n")

    def test_27_fit(self):
        print("Start fit-27 test\n")

        x_train_int = tuple([42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                     33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                     43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                     13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                     10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                     37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                     49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                     26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                     18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                     36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                     85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                     81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                     79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                     87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                     70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                     82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                     71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                     88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                     87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                     72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        x_train_float = tuple([float(x) for x in x_train_int])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(x_train_float,
                                                                                                                    self.y_train,
                                                                                                                    mode = self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(x_train_float,
                                                                    self.y_train,
                                                                    mode = self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-27 test\n")

    def test_28_fit(self):
        print("Start fit-28 test\n")

        y_train_int = [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        y_train_float = [float(y) for y in y_train_int]

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    y_train_float,
                                                                                                                    mode = self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    y_train_float,
                                                                    mode = self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-28 test\n")

    def test_29_fit(self):

        print("Start fit-29 test\n")

        y_train_int = tuple([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                       1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        y_train_float = tuple([float(y) for y in y_train_int])

        if self.mode == logging.DEBUG:
            pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.univariate_lr_test.fit(self.x_train,
                                                                                                                    y_train_float,
                                                                                                                    mode = self.mode)
        else:
            pred_slope, pred_intercept = self.univariate_lr_test.fit(self.x_train,
                                                                    y_train_float,
                                                                    mode = self.mode)
        actual_slope = -0.3921094504917619
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = 22.78930685840734
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-29 test\n")

    def test_1_init(self):

        print("Start pred-1 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope="hello",
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)

        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-1 test\n")

    def test_2_init(self):

        print("Start pred-2 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=0,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)

        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-2 test\n")

    def test_3_init(self):

        print("Start pred-3 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=0.0,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-3 test\n")

    def test_4_init(self):

        print("Start pred-4 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=[0],
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)

        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-4 test\n")

    def test_5_init(self):

        print("Start pred-5 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=tuple([0]),
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-5 test\n")

    def test_6_init(self):

        print("Start pred-6 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=np.array([0]),
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-6 test\n")

    def test_7_init(self):

        print("Start pred-6 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept='hello',
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-7 test\n")

    def test_8_init(self):

        print("Start pred-8 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=0,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-8 test\n")

    def test_9_init(self):

        print("Start pred-9 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=0.0,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-9 test\n")

    def test_10_init(self):

        print("Start pred-10 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=[0],
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-10 test\n")

    def test_11_init(self):

        print("Start pred-11 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=tuple([0]),
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-11 test\n")

    def test_12_init(self):

        print("Start pred-12 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=np.array([0]),
                                                                                           learning_rate=0.00001,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-12 test\n")

    def test_13_init(self):

        print("Start pred-12 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate='hello',
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-13 test\n")

    def test_14_init(self):

        print("Start pred-14 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-14 test\n")

    def test_15_init(self):

        print("Start pred-15 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.0,
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-15 test\n")

    def test_16_init(self):

        print("Start pred-16 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=[0],
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-16 test\n")

    def test_17_init(self):

        print("Start pred-17 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=tuple([0]),
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-17 test\n")

    def test_18_init(self):

        print("Start pred-18 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=np.array([0]),
                                                                                           epochs=100000)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-18 test\n")


    def test_19_init(self):

        print("Start pred-19 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs='hello')
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-19 test\n")

    def test_20_init(self):

        print("Start pred-20 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=0)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-20 test\n")
    def test_21_init(self):

        print("Start pred-21 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=0.0)
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-21 test\n")

    def test_22_init(self):

        print("Start pred-22 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=[0])
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-22 test\n")

    def test_23_init(self):

        print("Start pred-23 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=tuple([0]))
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-23 test\n")

    def test_24_init(self):

        print("Start pred-24 test\n")

        # creating linear_regression object
        univariate_lr_test = univariate_logistic_regression.univariate_logistic_regression(slope=1,
                                                                                           intercept=1,
                                                                                           learning_rate=0.00001,
                                                                                           epochs=np.array([0]))
        print("univariate_lr_test: {univariate_lr_test}")
        print("\nFinish init-24 test\n")

    def test_1_pred(self):

        print("Start pred-1 test\n")

        pred_yhat = self.univariate_lr_test.pred(self.x_train,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-1 test\n")

    def test_2_pred(self):

        print("Start pred-2 test\n")

        pred_yhat = self.univariate_lr_test.pred("hello",
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-2 test\n")
    def test_3_pred(self):

        print("Start pred-3 test\n")

        pred_yhat = self.univariate_lr_test.pred(0,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-3 test\n")

    def test_4_pred(self):

        print("Start pred-4 test\n")

        pred_yhat = self.univariate_lr_test.pred(0.0,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-4 test\n")

    def test_5_pred(self):

        print("Start pred-5 test\n")

        pred_yhat = self.univariate_lr_test.pred([0],
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-5 test\n")

    def test_6_pred(self):

        print("Start pred-6 test\n")

        pred_yhat = self.univariate_lr_test.pred(tuple([0]),
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-6 test\n")

    def test_7_pred(self):

        print("Start pred-7 test\n")

        x_train_int = [42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        pred_yhat = self.univariate_lr_test.pred(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-7 test\n")

    def test_8_pred(self):

        print("Start pred-8 test\n")

        x_train_int = tuple([42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_yhat = self.univariate_lr_test.pred(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-8 test\n")

    def test_9_pred(self):

        print("Start pred-9 test\n")

        x_train_int = ['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        pred_yhat = self.univariate_lr_test.pred(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-9 test\n")
    def test_10_pred(self):

        print("Start pred-10 test\n")

        x_train_int = tuple(['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_yhat = self.univariate_lr_test.pred(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-10 test\n")

    def test_11_pred(self):

        print("Start pred-11 test\n")

        x_train_int = np.array(['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_yhat = self.univariate_lr_test.pred(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-11 test\n")

    def test_12_pred(self):

        print("Start pred-12 test\n")

        pred_yhat = self.univariate_lr_test.pred(np.array([0]),
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_yhat, self.y_pred)

        print("\nFinish pred-12 test\n")

    def test_1_pred_prob(self):

        print("Start pred_prob-1 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob(self.x_train,
                                                        slope=-0.3921094504917619,
                                                        intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-1 test\n")

    def test_2_pred_prob(self):

        print("Start pred_prob-2 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob('hello',
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-2 test\n")

    def test_3_pred_prob(self):

        print("Start pred_prob-3 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob(0,
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-3 test\n")

    def test_4_pred_prob(self):

        print("Start pred_prob-4 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob(0.0,
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-4 test\n")

    def test_5_pred_prob(self):

        print("Start pred_prob-5 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob([0],
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-5 test\n")

    def test_6_pred_prob(self):

        print("Start pred_prob-6 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob(tuple([0]),
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-6 test\n")

    def test_7_pred_prob(self):

        print("Start pred_prob-7 test\n")

        x_train_int = [42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        pred_prob_yhat = self.univariate_lr_test.pred_prob(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-7 test\n")

    def test_8_pred_prob(self):

        print("Start pred_prob-8 test\n")

        x_train_int = tuple([42.54174696, 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                             33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                             43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                             13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                             10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                             37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                             49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                             26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                             18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                             36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                             85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                             81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                             79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                             87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                             70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                             82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                             71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                             88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                             87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                             72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_prob_yhat = self.univariate_lr_test.pred_prob(x_train_int,
                                                slope=-0.3921094504917619,
                                                intercept=22.78930685840734)

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-8 test\n")

    def test_9_pred_prob(self):

        print("Start pred_prob-9 test\n")

        x_train_int = ['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                       13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                       82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                       71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523]

        pred_prob_yhat = self.univariate_lr_test.pred_prob(x_train_int,
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-9 test\n")

    def test_10_pred_prob(self):

        print("Start pred_prob-10 test\n")

        x_train_int = tuple(['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                       13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                       82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                       71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_prob_yhat = self.univariate_lr_test.pred_prob(x_train_int,
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-10 test\n")

    def test_11_pred_prob(self):

        print("Start pred_prob-11 test\n")

        x_train_int = np.array(['xyz', 16.5657875, 27.4220827, 42.86694542, 26.33604129,
                       33.94325895, 30.32185093, 21.24402881, 23.04851584, 46.50530449,
                       43.91413153, 27.91093155, 19.68190603, 47.024089, 22.65059986,
                       13.00476328, 10.2531771, 26.71415627, 36.56821668, 48.75500207,
                       10.48419193, 42.39608402, 23.93262524, 33.15620189, 34.85873111,
                       37.13988412, 25.6300619, 49.37105973, 26.04078426, 13.92191395,
                       49.93331999, 32.71308919, 25.42231186, 39.98775064, 30.94362924,
                       26.93887616, 10.45597313, 25.68370408, 20.76164422, 48.32269811,
                       18.46609927, 13.54151022, 47.12960729, 13.22169817, 41.23138269,
                       36.90841835, 32.37116958, 37.00922821, 30.55495251, 20.50071673,
                       85.84185804, 76.58300638, 82.16671658, 81.62365727, 82.90023151,
                       81.88745073, 81.38352677, 88.94190514, 74.75010778, 81.50443846,
                       79.78983199, 89.88739019, 82.24502463, 73.91542633, 71.32368807,
                       87.25136131, 70.36273723, 83.27299981, 80.21626445, 76.04371743,
                       70.17175933, 78.09313369, 88.37201905, 77.3322709, 86.33721108,
                       82.00979265, 75.28765789, 85.1550368, 74.86287242, 71.82513828,
                       71.21482288, 85.2373968, 71.56759081, 80.99125295, 76.93225106,
                       88.01730941, 85.03826558, 70.54311964, 86.65357844, 72.18957558,
                       87.6918872, 70.66837137, 76.19374054, 87.26463937, 75.94397752,
                       72.23677043, 77.48037136, 82.80284985, 77.14019787, 82.14418523])

        pred_prob_yhat = self.univariate_lr_test.pred_prob(x_train_int,
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-11 test\n")

    def test_12_pred_prob(self):

        print("Start pred_prob-12 test\n")

        pred_prob_yhat = self.univariate_lr_test.pred_prob(np.array([0]),
                                                           slope=-0.3921094504917619,
                                                           intercept=22.78930685840734)

        print(f"pred_prob_yhat: {pred_prob_yhat}")

        self.assertEqual(pred_prob_yhat, self.y_pred_prob)

        print("\nFinish pred_prob-12 test\n")

if __name__ == '__main__':
    unittest.main()
