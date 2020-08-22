import os
''' 
* In this file, we will call the overall training process of FL to calculate the average accuracy
* As a result, os.system() will be used
'''


def main():
    average_time = 1
    retrain_update_time = 2
    for i in range(average_time):
        for K in [100]:
            for dim in [10000]:
                os.system("python \"Base Model/Base_model.py\" -K " +
                          str(K)+" -D "+str(dim))
                # # Windows10
                # # os.chdir(os.path.dirname(os.path.abspath(__file__)))
                # # Ubuntu
                # # os.chdir(os.path.dirname(
                # #     os.path.dirname(os.path.abspath(__file__))))
                os.system("python client_training.py")
                os.system("python ./Global_model/global_model.py")
                for retrain_epoch in range(retrain_update_time):
                    os.system("python client_retraining.py")
                    os.system(
                        "python ./Global_model/global_model.py {}".format(retrain_epoch+1))


if __name__ == "__main__":
    main()
