import os
''' 
* In this file, we will call the overall training process of FL to calculate the average accuracy
* As a result, os.system() will be used
'''


def main():
    average_time = 5
    for i in range(average_time):
        for K in [20, 40, 80, 100, 120]:
            for dim in [1000, 2000, 5000, 10000]:
                os.chdir(os.path.join(os.path.dirname(__file__), 'Base Model'))
                os.system("python \"./Base_model.py\" -K " +
                          str(K)+" -D "+str(dim))
                # Windows10
                os.chdir(os.path.dirname(os.path.abspath(__file__)))
                # Ubuntu
                # os.chdir(os.path.dirname(
                # os.path.dirname(os.path.abspath(__file__))))
                os.system("python client_retraining.py")
                os.system("python ./Global_model/re_global_model.py")


if __name__ == "__main__":
    main()
