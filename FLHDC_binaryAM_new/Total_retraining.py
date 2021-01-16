import os
''' 
* In this file, we will call the overall training process of FL to calculate the average accuracy
* As a result, os.system() will be used
'''


def main():
    average_time = 3
    retrain_update_time = 50
    for i in range(average_time):
        for K in [20]:
            for dim in [10000]:
                for lr_max in [3, 7, 10, 20]:
                    os.system("python \"Base Model/Base_model.py\" -K " +
                              str(K)+" -D "+str(dim))
                    os.system("python client_training.py")
                    os.system(
                        "python ./Global_model/global_model.py {}".format(lr_max))
                    os.system("python ./Global_model/plot_histogram.py")
                    for retrain_epoch in range(retrain_update_time):
                        os.system("python client_retraining.py")
                        os.system(
                            "python ./Global_model/global_model.py {} {}".format(lr_max, retrain_epoch+1))
                        os.system(
                            "python ./Global_model/plot_histogram.py {}".format(retrain_epoch+1))


if __name__ == "__main__":
    main()
