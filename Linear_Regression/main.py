import unittest
import logging
import numpy as np
import linear_regression as linear_regression

class test(unittest.TestCase):

    # creating linear_regression object
    linear_test = linear_regression.linear_regression()

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

    y_hat = [-122669.93370857342,
             -30571.789469245246,
             -13121.61477126728,
             -39296.87681823423,
             -188592.81590093466,
             -64502.68471531352,
             -161448.09970408003,
             -149814.64990542806,
             -56747.051516212196,
             -181806.63685172098,
             -88739.03846250514,
             -25724.51871980692,
             -21846.70212025626,
             -65472.13886520118,
             -171142.64120295667,
             -184714.99930138397,
             -95525.2175117188,
             -187623.361751047,
             -148845.1957555404,
             -107158.66731037077,
             -86800.13016272981,
             -24755.064569919257,
             -164356.46215374302,
             -57716.50566609987,
             -103280.85071082012,
             -163387.00800385536,
             -133333.92935733774,
             -96494.67166160647,
             -132364.47520745007,
             -91647.40091216813,
             -12152.160621379613,
             -67411.04716497651,
             -82922.31356317915,
             -98433.5799613818,
             -31541.24361913291,
             -23785.61042003159,
             -162417.5538539677,
             -59655.413965875196,
             -143028.47085621438,
             -169203.73290318134,
             -36388.51436857123,
             -5365.98157216596,
             -140120.1084065514,
             -126547.75030812407,
             -183745.5451514963,
             -151753.5582052034,
             -156600.8289546417,
             -44144.14756767255,
             -53838.6890665492,
             -7304.889871941289,
             -165325.91630363069,
             -174051.00365261966,
             -186653.9076011593,
             -192470.6325004853,
             -43174.69341778489,
             -40266.330968121896,
             -144967.3791559897,
             -3427.0732723906294,
             -135272.83765711306,
             -120731.02540879809,
             -131395.0210575624,
             -73227.77206430251,
             -116853.20880924743,
             -138181.20010677606,
             -172112.09535284434,
             -66441.59301508885,
             -81952.85941329149,
             -141089.56255643905,
             -100372.48826115712,
             -50930.32661688621,
             -99403.03411126946,
             -84861.22186295448,
             -109097.5756101461,
             -20877.247970368597,
             -16999.431370817936,
             -90677.94676228047,
             -68380.50131486417,
             -101341.94241104479,
             -83891.76771306682,
             -518.7108227276348,
             -48021.96416722322,
             -193440.08665037298,
             -11182.706471491949,
             -119761.57125891042,
             -97464.12581149413,
             -37357.9685184589,
             -173081.549502732,
             -49960.872466998546,
             -62563.77641553819,
             -14091.068921154943,
             -34449.6060687959,
             -42205.239267897225,
             -74197.22621419017,
             -92616.8550620558,
             -89708.4926123928,
             -51899.780766773874,
             -58685.95981598753,
             -80983.40526340382,
             -145936.8333058774,
             -166295.37045351835,
             -94555.76336183114,
             -175020.45780250733,
             -79044.4969636285,
             -77105.58866385317,
             -159509.1914043047,
             -102311.39656093245,
             -80013.95111351616,
             -155631.37480475404,
             -167264.824603406,
             -71288.86376452717,
             -113944.84635958442,
             -175989.911952395,
             -117822.6629591351,
             -189562.27005082232,
             -157570.28310452937,
             -48991.41831711088,
             -105219.75901059544,
             -29602.335319357582,
             -26693.972869694586,
             -121700.47955868575,
             -22816.156270143925,
             -129456.11275778706,
             -182776.09100160864,
             -110067.02976003377,
             -177928.82025217032,
             -17968.8855207056,
             -176959.36610228266,
             -76136.1345139655,
             -104250.30486070778,
             -170173.187053069,
             -168234.27875329368,
             -54808.14321643687,
             -87769.58431261747,
             -179867.72855194565,
             -15060.52307104261,
             -28632.881169469914,
             -178898.27440205798,
             -112975.39220969676,
             -190531.72420070998,
             -185684.45345127163,
             -60624.86811576286,
             -63533.23056542585,
             -106189.21316048311,
             -160478.64555419236,
             -118792.11710902276,
             -41235.78511800956,
             -4396.527422278295,
             -69349.95546475184,
             -147875.74160565273,
             -55777.59736632453,
             -33480.15191890824,
             -130425.56690767473,
             -139150.65425666372,
             -112005.9380598091,
             -8274.344021828954,
             -108128.12146025844,
             -150784.10405531572,
             -10213.252321604285,
             -93586.30921194346,
             -27663.42701958225,
             -46083.05586744788,
             -134303.3835072254,
             -125578.29615823641,
             -136242.29180700073,
             -1488.1649726152996,
             -154661.92065486638,
             -32510.697769020575,
             -191501.17835059765,
             -2457.6191225029647,
             -114914.3005094721,
             -152723.01235509105,
             -115883.75465935977,
             -153692.4665049787,
             -128486.6586078994,
             -70319.4096146395,
             -9243.798171716619,
             -143997.92500610204,
             -38327.42266834657,
             -146906.28745576506,
             -75166.68036407784,
             -6335.435722053624,
             -72258.31791441483,
             -18938.33967059327,
             -111036.48390992143,
             -19907.793820480932,
             -142059.0167063267,
             -78075.04281374083,
             -61594.322265650524,
             -180837.1827018333,
             -137211.7459568884,
             -35419.06021868357,
             -45113.60171756022,
             -85830.67601284214,
             -52869.23491666154,
             -158539.73725441704,
             -127517.20445801174,
             -47052.510017335546,
             -123639.38785846108,
             -16029.977220930274,
             -124608.84200834874]


    # It is a test case function to check the linear_regression.fit function - 1
    def test_1_fit(self):

        print("Start fit-1 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept
        print("\nFinish fit-1 test\n")

    def test_2_fit(self):

        print("Start fit-2 test\n")
        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(0,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-2 test\n")

    def test_3_fit(self):

        print("Start fit-3 test\n")
        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(0.0,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-3 test\n")

    def test_4_fit(self):

        print("Start fit-4 test\n")

        x_train = [126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                   131, 48, 127, 16, 128]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-4 test\n")

    def test_5_fit(self):

        print("Start fit-5 test\n")
        x_train = tuple([126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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


        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-5 test\n")

    def test_6_fit(self):

        print("Start fit-6 test\n")

        x_train = ['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                         131, 48, 127, 16, 128]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-6 test\n")

    def test_7_fit(self):
        print("Start fit-7 test\n")
        x_train = tuple(['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-7 test\n")

    def test_8_fit(self):

        print("Start fit-8 test\n")

        x_train = np.array(['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-8 test\n")

    def test_9_fit(self):

        print("Start fit-9 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit([0],
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-9 test\n")

    def test_10_fit(self):

        print("Start fit-10 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(tuple([0]),
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-10 test\n")

    def test_11_fit(self):

        print("Start fit-11 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-11 test\n")

    def test_12_fit(self):

        print("Start fit-12 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          0,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-12 test\n")

    def test_13_fit(self):

        print("Start fit-13 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          0.0,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-13 test\n")

    def test_14_fit(self):

        print("Start fit-14 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          [0],
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-14 test\n")

    def test_15_fit(self):

        print("Start fit-15 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          tuple([0]),
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-15 test\n")

    def test_16_fit(self):

        print("Start fit-16 test\n")

        y_train = [-127995, -29000, -11998, -41993, -204001, -63997, -166993,
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
                   -49991, -122997, -7010, -121010]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-16 test\n")

    def test_17_fit(self):

        print("Start fit-17 test\n")

        y_train = tuple([-127995, -29000, -11998, -41993, -204001, -63997, -166993,
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

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-17 test\n")

    def test_18_fit(self):

        print("Start fit-18 test\n")

        y_train = ['xyz', -29000, -11998, -41993, -204001, -63997, -166993,
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
                   -49991, -122997, -7010, -121010]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-18 test\n")

    def test_19_fit(self):

        print("Start fit-19 test\n")

        y_train = tuple(['xyz', -29000, -11998, -41993, -204001, -63997, -166993,
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

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-19 test\n")

    def test_20_fit(self):

        print("Start fit-20 test\n")

        y_train = np.array(['xyz', -29000, -11998, -41993, -204001, -63997, -166993,
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

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-20 test\n")

    def test_21_fit(self):

        print("Start fit-21 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode='hello')
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-21 test\n")

    def test_22_fit(self):

        print("Start fit-22 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=0)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-22 test\n")

    def test_23_fit(self):

        print("Start fit-23 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=0.0)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-23 test\n")

    def test_24_fit(self):

        print("Start fit-24 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=[0])
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-24 test\n")

    def test_25_fit(self):

        print("Start fit-25 test\n")

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          self.y_train,
                                                                                                          mode=tuple([0]))
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept

        print("\nFinish fit-25 test\n")


    def test_26_fit(self):

        print("Start fit-26 test\n")

        x_train_int = [126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                         131, 48, 127, 16, 128]

        x_train_float = [float(x) for x in x_train_int]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train_float,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept
        print("\nFinish fit-26 test\n")

    def test_27_fit(self):

        print("Start fit-27 test\n")

        x_train_int = [126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                       131, 48, 127, 16, 128]

        x_train_float = tuple([float(x) for x in x_train_int])

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(x_train_float,
                                                                                                          self.y_train,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept
        print("\nFinish fit-27 test\n")

    def test_28_fit(self):

        print("Start fit-28 test\n")

        y_train_int = [-127995, -29000, -11998, -41993, -204001, -63997, -166993,
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
                         -49991, -122997, -7010, -121010]

        y_train_float = [float(x) for x in y_train_int]

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train_float,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept
        print("\nFinish fit-28 test\n")

    def test_29_fit(self):

        print("Start fit-29 test\n")

        y_train_int = tuple([-127995, -29000, -11998, -41993, -204001, -63997, -166993,
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

        y_train_float = tuple([float(x) for x in y_train_int])

        pred_slope, pred_intercept, loss_history, slope_history, intercept_history = self.linear_test.fit(self.x_train,
                                                                                                          y_train_float,
                                                                                                          mode=self.mode)
        actual_slope = -969.4541498876649
        actual_slope = float(f"{actual_slope:.2f}")
        actual_intercept = -518.7108227276348
        actual_intercept = float(f"{actual_intercept:.2f}")

        pred_slope = float(f"{pred_slope:.2f}")
        pred_intercept = float(f"{pred_intercept:.2f}")

        self.assertEqual(pred_slope, actual_slope)  # slope
        self.assertEqual(pred_intercept, actual_intercept)  # intercept
        print("\nFinish fit-29 test\n")


    def test_1_init(self):
        print("Start init-1 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope='hello',
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-1 test\n")

    def test_2_init(self):
        print("Start init-2 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=0,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-2 test\n")

    def test_3_init(self):
        print("Start init-3 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=0.0,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-3 test\n")

    def test_4_init(self):
        print("Start init-4 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=[0],
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-2 test\n")

    def test_5_init(self):
        print("Start init-5 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=tuple([0]),
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-5 test\n")

    def test_6_init(self):
        print("Start init-6 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=np.array([0]),
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-6 test\n")

    def test_7_init(self):
        print("Start init-7 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept='hello',
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-7 test\n")

    def test_8_init(self):
        print("Start init-8 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=0,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-8 test\n")

    def test_9_init(self):
        print("Start init-9 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=0.0,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-9 test\n")

    def test_10_init(self):
        print("Start init-10 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=[0],
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-10 test\n")

    def test_11_init(self):
        print("Start init-11 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=tuple([0]),
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_i nit_test}")
        print("\nFinish init-11 test\n")

    def test_12_init(self):
        print("Start init-12 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=np.array([0]),
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-12 test\n")

    def test_13_init(self):
        print("Start init-13 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate='hello',
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-13 test\n")

    def test_14_init(self):
        print("Start init-14 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-14 test\n")

    def test_15_init(self):
        print("Start init-15 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.0,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-15 test\n")

    def test_16_init(self):
        print("Start init-16 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=[0],
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-16 test\n")

    def test_17_init(self):
        print("Start init-17 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=tuple([0]),
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-17 test\n")

    def test_18_init(self):
        print("Start init-18 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=np.array([0]),
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-18 test\n")

    def test_19_init(self):
        print("Start init-19 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs='hello')
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-19 test\n")

    def test_20_init(self):
        print("Start init-20 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=0)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-20 test\n")

    def test_21_init(self):
        print("Start init-21 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=0.0)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-21 test\n")

    def test_22_init(self):
        print("Start init-22 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=[0])
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-22 test\n")

    def test_23_init(self):
        print("Start init-23 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=tuple([0]))
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-23 test\n")

    def test_24_init(self):
        print("Start init-24 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=np.array([0]))
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-24 test\n")

    def test_25_init(self):
        print("Start init-25 test\n")

        # creating linear_regression object
        linear_init_test = linear_regression.linear_regression(slope=1,
                                                                intercept=1,
                                                                learning_rate=0.00001,
                                                                epochs=100000)
        print("linear_init_test: {linear_init_test}")
        print("\nFinish init-25 test\n")

    def test_1_pred(self):
        print("Start pred-1 test\n")

        pred_yhat = self.linear_test.pred(self.x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-1 test\n")

    def test_2_pred(self):
        print("Start pred-2 test\n")

        pred_yhat = self.linear_test.pred('hello')
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-2 test\n")

    def test_3_pred(self):
        print("Start pred-3 test\n")

        pred_yhat = self.linear_test.pred(0)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-3 test\n")

    def test_4_pred(self):
        print("Start pred-4 test\n")

        pred_yhat = self.linear_test.pred(0.0)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-4 test\n")

    def test_5_pred(self):
        print("Start pred-5 test\n")

        pred_yhat = self.linear_test.pred([0])
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-5 test\n")

    def test_6_pred(self):
        print("Start pred-6 test\n")

        pred_yhat = self.linear_test.pred(tuple([0]))
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-6 test\n")

    def test_7_pred(self):
        print("Start pred-7 test\n")

        x_train = [126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                   131, 48, 127, 16, 128]

        pred_yhat = self.linear_test.pred(x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-7 test\n")

    def test_8_pred(self):
        print("Start pred-8 test\n")

        x_train = tuple([126, 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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

        pred_yhat = self.linear_test.pred(x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-8 test\n")

    def test_9_pred(self):
        print("Start pred-9 test\n")

        x_train = ['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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
                   131, 48, 127, 16, 128]

        pred_yhat = self.linear_test.pred(x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-9 test\n")

    def test_10_pred(self):
        print("Start pred-10 test\n")

        x_train = tuple(['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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

        pred_yhat = self.linear_test.pred(x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-10 test\n")

    def test_11_pred(self):
        print("Start pred-11 test\n")

        x_train = np.array(['xyz', 31, 13, 40, 194, 66, 166, 154, 58, 187, 91, 26, 22,
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

        pred_yhat = self.linear_test.pred(x_train)
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-11 test\n")

    def test_12_pred(self):
        print("Start pred-12 test\n")

        pred_yhat = self.linear_test.pred(np.array([0]))
        self.assertEqual(pred_yhat, self.y_hat)  # slope

        print("\nFinish pred-12 test\n")


if __name__ == '__main__':
    unittest.main()
