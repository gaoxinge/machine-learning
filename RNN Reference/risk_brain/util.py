from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def draw_roc(actuals, predicts):
    """
    绘制roc曲线
    :param actual: 真实值
    :param predictions: 预测值
    :return:
    """
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actuals, predicts)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('ROC curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.9f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # test draw roc
    actual = [1, 1, 1, 0, 0, 0]
    predictions = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1]
    draw_roc(actual, predictions)


