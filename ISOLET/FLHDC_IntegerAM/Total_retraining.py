import os
''' 
* In this file, we will call the overall training process of FL to calculate the average accuracy
* As a result, os.system() will be used
'''


def main():
    average_time = 10
    retrain_update_time = 50
    for i in range(average_time):
        for K in [20]:
            for dim in [10000]:
                os.system("python3 \"Base Model/Base_model.py\" -K " +
                          str(K)+" -D "+str(dim))
                os.system("python3 client_training.py")
                os.system("python3 ./Global_model/global_model.py")
                for retrain_epoch in range(retrain_update_time):
                    os.system("python3 client_retraining.py")
                    os.system(
                        "python3 ./Global_model/global_model.py {}".format(retrain_epoch+1))


if __name__ == "__main__":
    main()
