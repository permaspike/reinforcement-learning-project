import os

from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd


def plot_history(dataLoader, networkName, history):
    plt.figure(figsize=(7,5))
    plt.plot(history['loss'], color='green')
    plt.plot(history['val_loss'], color='red')
    plt.title(networkName + " Loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),dataLoader.resultsDir, 'train_history_loss' + '.png'))
    plt.show()

    plt.figure(figsize=(7,5))
    plt.plot(history['accuracy'], color='green')
    plt.plot(history['val_accuracy'], color='red')
    plt.title(networkName + " Accuracy")
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),dataLoader.resultsDir, 'train_history_acc' +'.png'))
    plt.show()


def plot_actions_stats(dataLoader, networkName, actions, stats, filename):
    plt.bar(actions, height=stats, width=.6)
    plt.title('Actions Taken')
    plt.ylabel('number of times action was chosen')
    plt.xlabel('action name')
    # plt.savefig(os.path.join(dataLoader.resultsDir, 'actions_stats' + filename + networkName + dataLoader.datasetInfo +
    #                          '.png'))
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), dataLoader.resultsDir, filename+".png"))
    plt.show()


def plot_conf_matrix(dataLoader, networkName, conf_matrix, classes, alg):
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(7,5))
    plt.title(alg)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), dataLoader.resultsDir, 'conf_matrix' + alg +
                             '.png'))
    plt.show()

def print_classification_details(statController):
    print("accuracy: ", statController.accuracy)
    print("precision: ", statController.precision)
    print("recall: ", statController.recall)
    print("F1 score: ", statController.f1Score)
    print("report: ", statController.report)

def plot_misclassif(dataLoader, RLstats, alg):
    plt.bar(["Q-Learning", "Monte Carlo", "DQN", "No RL"], height=RLstats, width=.6, color=['green', 'green', 'green', 'red'])
    #plt.title(alg)
    plt.ylabel("Number of Misclassifications")
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), dataLoader.resultsDir, "misclassifications"+ alg + ".png"))
    plt.show()

