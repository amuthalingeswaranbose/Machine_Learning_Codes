try:
    import unittest
    import logging
    import numpy as np
    import linear_regression as linear_regression

except ImportError:
    print("please install required packages for import(unittest,numpy)...")

class test(unittest.TestCase):
    """
    the basic class inherits .TestCase
    """

    try:
        # create and configure logger
        logging.basicConfig(filename='debug.log', format="%(asctime)s %(message)s", filemode='w')

        # creating an object
        logger = logging.getLogger()

        # setting the threshold of logger to DEBUG
        logger.setLevel(logging.DEBUG)

        # mode
        mode = logging.DEBUG

        # initialize
        x_train = np.array([126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
                            67, 176, 190, 98, 193, 153, 110, 89, 25, 169, 59, 106, 168,
                            137, 99, 136, 94, 12, 69, 85, 101, 32, 24, 167, 61, 147,
                            174, 37, 5, 144, 130, 189, 156, 161, 45, 55, 7, 170, 179,
                            192, 198, 44, 41, 149, 3, 139, 124, 135, 75, 120, 142, 177,
                            68, 84, 145, 103, 52, 102, 87, 112, 21, 17, 93, 70, 104,
                            86, 0, 49, 199, 11, 123, 100, 38, 178, 51, 64, 14, 35,
                            43, 76, 95, 92, 53, 60, 83, 150, 171, 97, 180, 81, 79,
                            164, 105, 82, 160, 172, 73, 117, 181, 121, 195, 162, 50, 108,
                            30, 27, 125, 23, 133, 188, 113, 183, 18, 182, 78, 107, 175,
                            173, 56, 90, 185, 15, 29, 184, 116, 196, 191, 62, 65, 109,
                            165, 122, 42, 4, 71, 152, 57, 34, 134, 143, 115, 8, 111,
                            155, 10, 96, 28, 47, 138, 129, 140, 1, 159, 33, 197, 2,
                            118, 157, 119, 158, 132, 72, 9, 148, 39, 151, 77, 6, 74,
                            19, 114, 20, 146, 80, 63, 186, 141, 36, 46, 88, 54, 163,
                            131, 48, 127, 16, 128])

        y_train = np.array([-127995, -29000, -11998, -41993, -204001, -63997, -166993,
                            -162992, -56009, -177995, -90998, -16995, -22005, -74997,
                            -176997, -189001, -105007, -190005, -148998, -112996, -85006,
                            -23998, -170009, -66002, -110002, -178003, -133006, -89998,
                            -134003, -84993, -12999, -62010, -81000, -96991, -31992,
                            -17007, -177009, -68001, -151995, -176001, -33009, -12001,
                            -136999, -131005, -182003, -154996, -162003, -51008, -61009,
                            -16006, -168992, -179001, -202002, -201000, -38998, -47001,
                            -157006, -4004, -141996, -120003, -143000, -79992, -117992,
                            -134007, -173004, -63992, -83009, -147996, -96010, -50002,
                            -105995, -82009, -106000, -18007, -16001, -91000, -79997,
                            -112005, -77997, -8999, -43999, -205006, -9995, -129002,
                            -103001, -36999, -178998, -60000, -69003, -16999, -41006,
                            -46995, -76008, -105007, -93003, -54003, -57991, -81007,
                            -149995, -167994, -95008, -181007, -74004, -85991, -164005,
                            -100005, -90996, -162003, -172009, -66005, -127008, -174009,
                            -125995, -186997, -171998, -58994, -99005, -22003, -27996,
                            -126991, -30002, -137007, -181010, -108000, -181008, -18997,
                            -192001, -83002, -102999, -184999, -178005, -58996, -83007,
                            -185003, -9999, -36998, -194000, -123994, -189993, -190992,
                            -64997, -74994, -110000, -165005, -122009, -40992, -13999,
                            -73005, -145008, -57002, -29006, -128006, -147010, -120992,
                            1001, -118991, -154004, -18010, -101994, -35996, -38994,
                            -140010, -132009, -131991, -8001, -165992, -24002, -196000,
                            5001, -122991, -164003, -115994, -167008, -124003, -64992,
                            -12000, -142002, -29993, -153996, -68997, -9, -74995,
                            -17005, -116001, -30006, -138008, -72998, -59992, -178993,
                            -150003, -36006, -52006, -94000, -54008, -171005, -122005,
                            -49991, -122997, -7010, -121010])
    except ValueError:
        print("please initialize correct values and try again")

    # It is a test case function to check the linear_regression.fit function - 1
    def test_0_fit(self):

        try:
            print("Start fit-0 test\n")
            self.logger.debug("Start fit-0 test\n")

            # creating linear_regression object
            linear_test0 = linear_regression.linear_regression()

            if self.mode == logging.DEBUG:
                pred_slope, pred_intercept, loss_history, slope_history, intercept_history = linear_test0.fit(self.x_train, self.y_train, mode=self.mode)
                self.logger.debug(f"pred_slope: {pred_slope}, pred_intercept: {pred_intercept}, loss_history: {loss_history}, slope_history: {slope_history}, intercept_history: {intercept_history}")
            else:
                pred_slope, pred_intercept, loss_history, slope_history, intercept_history = linear_test0.fit(self.x_train, self.y_train, mode=self.mode)
                self.logger.debug(f"pred_slope: {pred_slope}, pred_intercept: {pred_intercept}")

            actual_slope = -969.4541498876649
            actual_slope = float(f"{actual_slope:.2f}")
            actual_intercept = -518.7108227276348
            actual_intercept = float(f"{actual_intercept:.2f}")

            pred_slope = float(f"{pred_slope:.2f}")
            pred_intercept = float(f"{pred_intercept:.2f}")

            self.assertEqual(pred_slope, actual_slope) # slope
            self.assertEqual(pred_intercept, actual_intercept) # intercept

            print("\nFinish fit-0 test\n")
            self.logger.debug("\nFinish fit-0 test\n")

        except ValueError:
            print("please check your test values and try again")


if __name__ == '__main__':
    unittest.main() 
