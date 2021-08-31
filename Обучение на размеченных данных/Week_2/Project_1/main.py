import numpy as np
from matplotlib import pyplot as plt
# import seaborn
from sklearn.metrics import precision_score, recall_score, accuracy_score, precision_recall_curve
from sklearn.metrics import f1_score, log_loss, roc_curve, roc_auc_score

def scatter(actual, predicted, T):
    plt.scatter(actual, predicted)
    plt.xlabel("Labels")
    plt.ylabel("Predicted probabilities")
    plt.plot([-0.2, 1.2], [T, T])
    plt.axis([-0.1, 1.1, -0.1, 1.1])
def many_scatters(actuals, predicteds, Ts, titles, shape):
    plt.figure(figsize=(shape[1] * 5, shape[0] * 5))
    i = 1
    for actual, predicted, T, title in zip(actuals, predicteds, Ts, titles):
        ax = plt.subplot(shape[0], shape[1], i)
        ax.set_title(title)
        i += 1
        scatter(actual, predicted, T)

def write_answer_1(precision_1, recall_1, precision_10, recall_10, precision_11, recall_11):
    answers = [precision_1, recall_1, precision_10, recall_10, precision_11, recall_11]
    with open("pa_metrics_problem1.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))
def write_answer_2(k_1, k_10, k_11):
    answers = [k_1, k_10, k_11]
    with open("pa_metrics_problem2.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))
def write_answer_3(wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11):
    answers = [wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11]
    with open("pa_metrics_problem3.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))
def write_answer_4(T_0, T_1, T_2, T_0r, T_1r, T_10, T_11):
    answers = [T_0, T_1, T_2, T_0r, T_1r, T_10, T_11]
    with open("pa_metrics_problem4.txt", "w") as fout:
        fout.write(" ".join([str(num) for num in answers]))

def weighted_log_loss(actual, predicted):
    w_r1 = 0.3
    w_r2 = 0.7
    n = len(actual)
    vec = w_r1 * actual * np.log(predicted) + w_r2 * (1 - actual) * np.log(1 - predicted)
    res = -np.sum(vec) / n
    return res

actual_0 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          1.,  1.,  1., 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
predicted_0 = np.array([ 0.19015288,  0.23872404,  0.42707312,  0.15308362,  0.2951875 ,
            0.23475641,  0.17882447,  0.36320878,  0.33505476,  0.202608  ,
            0.82044786,  0.69750253,  0.60272784,  0.9032949 ,  0.86949819,
            0.97368264,  0.97289232,  0.75356512,  0.65189193,  0.95237033,
            0.91529693,  0.8458463 ])

plt.figure(figsize=(5, 5))
scatter(actual_0, predicted_0, 0.5)
plt.show()

actual_1 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                    0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                    1.,  1.,  1.,  1.])
predicted_1 = np.array([ 0.41310733,  0.43739138,  0.22346525,  0.46746017,  0.58251177,
            0.38989541,  0.43634826,  0.32329726,  0.01114812,  0.41623557,
            0.54875741,  0.48526472,  0.21747683,  0.05069586,  0.16438548,
            0.68721238,  0.72062154,  0.90268312,  0.46486043,  0.99656541,
            0.59919345,  0.53818659,  0.8037637 ,  0.272277  ,  0.87428626,
            0.79721372,  0.62506539,  0.63010277,  0.35276217,  0.56775664])
actual_2 = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
predicted_2 = np.array([ 0.07058193,  0.57877375,  0.42453249,  0.56562439,  0.13372737,
            0.18696826,  0.09037209,  0.12609756,  0.14047683,  0.06210359,
            0.36812596,  0.22277266,  0.79974381,  0.94843878,  0.4742684 ,
            0.80825366,  0.83569563,  0.45621915,  0.79364286,  0.82181152,
            0.44531285,  0.65245348,  0.69884206,  0.69455127])

many_scatters([actual_0, actual_1, actual_2], [predicted_0, predicted_1, predicted_2],
              [0.5, 0.5, 0.5], ["Perfect", "Typical", "Awful algorithm"], (1, 3))
plt.show()

actual_0r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
predicted_0r = np.array([ 0.23563765,  0.16685597,  0.13718058,  0.35905335,  0.18498365,
            0.20730027,  0.14833803,  0.18841647,  0.01205882,  0.0101424 ,
            0.10170538,  0.94552901,  0.72007506,  0.75186747,  0.85893269,
            0.90517219,  0.97667347,  0.86346504,  0.72267683,  0.9130444 ,
            0.8319242 ,  0.9578879 ,  0.89448939,  0.76379055])
actual_1r = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
            1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
predicted_1r = np.array([ 0.13832748,  0.0814398 ,  0.16136633,  0.11766141,  0.31784942,
            0.14886991,  0.22664977,  0.07735617,  0.07071879,  0.92146468,
            0.87579938,  0.97561838,  0.75638872,  0.89900957,  0.93760969,
            0.92708013,  0.82003675,  0.85833438,  0.67371118,  0.82115125,
            0.87560984,  0.77832734,  0.7593189,  0.81615662,  0.11906964,
            0.18857729])

many_scatters([actual_0, actual_1, actual_0r, actual_1r],
              [predicted_0, predicted_1, predicted_0r, predicted_1r],
              [0.5, 0.5, 0.5, 0.5],
              ["Perfect careful", "Typical careful", "Perfect risky", "Typical risky"],
              (2, 2))
plt.show()

actual_10 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                1.,  1.,  1.])
predicted_10 = np.array([ 0.29340574, 0.47340035,  0.1580356 ,  0.29996772,  0.24115457,  0.16177793,
                         0.35552878,  0.18867804,  0.38141962,  0.20367392,  0.26418924, 0.16289102,
                         0.27774892,  0.32013135,  0.13453541, 0.39478755,  0.96625033,  0.47683139,
                         0.51221325,  0.48938235, 0.57092593,  0.21856972,  0.62773859,  0.90454639,  0.19406537,
                         0.32063043,  0.4545493 ,  0.57574841,  0.55847795 ])
actual_11 = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
predicted_11 = np.array([ 0.35929566, 0.61562123,  0.71974688,  0.24893298,  0.19056711,  0.89308488,
            0.71155538,  0.00903258,  0.51950535,  0.72153302,  0.45936068,  0.20197229,  0.67092724,
                         0.81111343,  0.65359427,  0.70044585,  0.61983513,  0.84716577,  0.8512387 ,
                         0.86023125,  0.7659328 ,  0.70362246,  0.70127618,  0.8578749 ,  0.83641841,
                         0.62959491,  0.90445368])

many_scatters([actual_1, actual_10, actual_11], [predicted_1, predicted_10, predicted_11],
              [0.5, 0.5, 0.5], ["Typical", "Avoids FP", "Avoids FN"], (1, 3))
plt.show()

T = 0.5
print("Алгоритмы, разные по качеству:")
for actual, predicted, descr in zip([actual_0, actual_1, actual_2],
                                    [predicted_0 > T, predicted_1 > T, predicted_2 > T],
                                    ["Perfect:", "Typical:", "Awful:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
        recall_score(actual, predicted), ";",\
        "accuracy =", accuracy_score(actual, predicted))
print()
print("Осторожный и рискующий алгоритмы:")
for actual, predicted, descr in zip([actual_1, actual_1r],
                                    [predicted_1 > T, predicted_1r > T],
                                    ["Typical careful:", "Typical risky:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
        recall_score(actual, predicted), ";",\
        "accuracy =", accuracy_score(actual, predicted))
print()
print("Разные склонности алгоритмов к ошибкам FP и FN:")
for actual, predicted, descr in zip([actual_10, actual_11],
                                    [predicted_10 > T, predicted_11 > T],
                                    ["Avoids FP:", "Avoids FN:"]):
    print(descr, "precision =", precision_score(actual, predicted), "recall =", \
        recall_score(actual, predicted), ";",\
        "accuracy =", accuracy_score(actual, predicted))
plt.show()

precs = []
recs = []
threshs = []
labels = ["Typical", "Avoids FP", "Avoids FN"]
for actual, predicted in zip([actual_1, actual_10, actual_11],
                                    [predicted_1, predicted_10, predicted_11]):
    prec, rec, thresh = precision_recall_curve(actual, predicted)
    precs.append(prec)
    recs.append(rec)
    threshs.append(thresh)
plt.figure(figsize=(15, 5))
for i in range(3):
    ax = plt.subplot(1, 3, i+1)
    plt.plot(threshs[i], precs[i][:-1], label="precision")
    plt.plot(threshs[i], recs[i][:-1], label="recall")
    plt.xlabel("threshold")
    ax.set_title(labels[i])
    plt.legend()
plt.show()

############### Programming assignment: problem 1 ###############
actuals = [actual_1, actual_10, actual_11]
predictions = [predicted_1, predicted_10, predicted_11]
inds = ['1', '10', '11']

T = 0.65
precisions = []
recalls = []
for actual, prediction in zip(actuals, predictions):
    tp = 0.
    tp_fp = 0.
    for var in enumerate(prediction):
        if var[1] > T:
            tp_fp += 1
            if actual[var[0]] == 1:
                tp += 1
    precisions.append(tp / tp_fp)
    recalls.append(tp / sum(actual))

for i in range(len(inds)):
    print('precision_{} = {}'.format(inds[i], precisions[i]))
    print('recall_{} = {}'.format(inds[i], recalls[i]))

# precision = TP/(TP + FP); sum(>T)
# recall = TP/(TP + FN); TP + FN = sum

precision_1 = 1.0
recall_1 = 0.4666666666666667
precision_10 = 1.0
recall_10 = 0.13333333333333333
precision_11 = 0.6470588235294118
recall_11 = 0.8461538461538461


write_answer_1(precision_1, recall_1, precision_10, recall_10, precision_11, recall_11)

T = 0.5
print("Разные склонности алгоритмов к ошибкам FP и FN:")
for actual, predicted, descr in zip([actual_1, actual_10, actual_11],
                                    [predicted_1 > T, predicted_10 > T, predicted_11 > T],
                                    ["Typical:", "Avoids FP:", "Avoids FN:"]):
    print(descr, "f1 =", f1_score(actual, predicted))
plt.show()

############### Programming assignment: problem 2 ###############
ks = []
precisions = []
recalls = []
Tmass = np.arange(0.1, 1.0, 0.1)

for actual, prediction in zip(actuals, predictions):
    f_1s = []
    for T in Tmass:
        tp = 0.
        tp_fp = 0.
        for var in enumerate(prediction):
            if var[1] > T:
                tp_fp += 1
                if actual[var[0]] == 1:
                    tp += 1
        precision = (tp / tp_fp)
        recall = tp / sum(actual)
        f_1s.append(2 * precision * recall / (precision + recall))
    print(max(f_1s))
    ks.append(int((Tmass[np.argmax(f_1s)]) * 10))

for i in range(len(inds)):
    print('k_{} = {}'.format(inds[i], ks[i]))

k_1 = 5
k_10 = 3
k_11 = 6

many_scatters([actual_1, actual_10, actual_11], [predicted_1, predicted_10, predicted_11],
              np.array(ks)*0.1, ["Typical", "Avoids FP", "Avoids FN"], (1, 3))

write_answer_2(k_1, k_10, k_11)

print("Алгоритмы, разные по качеству:")
for actual, predicted, descr in zip([actual_0, actual_1, actual_2],
                                    [predicted_0, predicted_1, predicted_2],
                                    ["Perfect:", "Typical:", "Awful:"]):
    print(descr, log_loss(actual, predicted))
print()
print("Осторожный и рискующий алгоритмы:")
for actual, predicted, descr in zip([actual_0, actual_0r, actual_1, actual_1r],
                                    [predicted_0, predicted_0r, predicted_1, predicted_1r],
                                    ["Ideal careful", "Ideal risky", "Typical careful:", "Typical risky:"]):
    print(descr, log_loss(actual, predicted))
print()
print("Разные склонности алгоритмов к ошибкам FP и FN:")
for actual, predicted, descr in zip([actual_10, actual_11],
                                    [predicted_10, predicted_11],
                                    ["Avoids FP:", "Avoids FN:"]):
    print(descr, log_loss(actual, predicted))
plt.show()


############### Programming assignment: problem 3 ###############

actuals = [actual_0, actual_1, actual_2, actual_0r, actual_1r, actual_10, actual_11]
predictions = [predicted_0, predicted_1, predicted_2, predicted_0r, predicted_1r, predicted_10, predicted_11]
inds = ['0', '1', '2', '0r', '1r', '10', '11']

wlls = []
for actual, prediction in zip(actuals, predictions):
    wlls.append(weighted_log_loss(actual, prediction))

for i in range(len(inds)):
    print('wll_{} = {}'.format(inds[i], wlls[i]))

wll_0 = 0.13125461813899453
wll_1 = 0.23013509212543612
wll_2 = 0.7350790493831211
wll_0r = 0.0841757752539052
wll_1r = 0.33544780012734865
wll_10 = 0.23785261402637708
wll_11 = 0.35866593961517557


write_answer_3(wll_0, wll_1, wll_2, wll_0r, wll_1r, wll_10, wll_11)


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
aucs = ""
for actual, predicted, descr in zip([actual_0, actual_1, actual_2],
                                    [predicted_0, predicted_1, predicted_2],
                                    ["Perfect", "Typical", "Awful"]):
    fpr, tpr, thr = roc_curve(actual, predicted)
    plt.plot(fpr, tpr, label=descr)
    aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.subplot(1, 3, 2)
for actual, predicted, descr in zip([actual_0, actual_0r, actual_1, actual_1r],
                                    [predicted_0, predicted_0r, predicted_1, predicted_1r],
                                    ["Ideal careful", "Ideal Risky", "Typical careful", "Typical risky"]):
    fpr, tpr, thr = roc_curve(actual, predicted)
    aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
    plt.plot(fpr, tpr, label=descr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.subplot(1, 3, 3)
for actual, predicted, descr in zip([actual_1, actual_10, actual_11],
                                    [predicted_1, predicted_10, predicted_11],
                                    ["Typical", "Avoids FP", "Avoids FN"]):
    fpr, tpr, thr = roc_curve(actual, predicted)
    aucs += descr + ":%3f"%roc_auc_score(actual, predicted) + " "
    plt.plot(fpr, tpr, label=descr)
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc=4)
plt.axis([-0.1, 1.1, -0.1, 1.1])
print(aucs)
plt.show()

############### Programming assignment: problem 4 ###############
inds = ['0', '1', '2', '0r', '1r', '10', '11']
Ts = []
for actual, prediction in zip(actuals, predictions):
    fpr, tpr, thr = roc_curve(actual, prediction)
    dist_s = (1-tpr)**2 + (0-fpr)**2
    Ts.append(thr[np.argmin(dist_s)])

for i in range(len(inds)):
    print('T_{} = {}'.format(inds[i], Ts[i]))

T_0 = 0.60272784
T_1 = 0.53818659
T_2 = 1.94843878
T_0r = 0.72007506
T_1r = 0.67371118
T_10 = 0.39478755
T_11 = 0.70044585


write_answer_4(T_0, T_1, T_2, T_0r, T_1r, T_10, T_11)