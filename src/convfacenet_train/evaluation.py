import sys
import time

from .utils import euclidean_distance, findCosineDistance
import pandas as pd
import os


def model_test(features_vectors_dict, dataset_frame, threshold=0.5, results_path=None,distance_method=None ,classifier=None, cuda=False):
    if distance_method is None:
        distance_method=euclidean_distance
    error = 0
    data_cnt = dataset_frame.count()[0]
    cnt = 0.0
    # row --> predicted Class  , cols --> true Class
    confusion_matrix = [[0, 0], [0, 0]]
    false_positive_rows = []
    false_negative_rows = []
    total_dist_same_persons = 0.0
    total_dist_diff_persons = 0.0
    avg_dist_same_persons = 0.0
    avg_dist_diff_persons = 0.0
    # 1--> two faces are for same person 0--> two faces are for different persons
    time_sum = 0.0
    for _, row in dataset_frame.iterrows():
        ts = time.time()

        img1_path = row[0]
        img2_path = row[1]

        emp1 = features_vectors_dict[img1_path]
        emp2 = features_vectors_dict[img2_path]

        if classifier is None:
            result = distance_method(emp1, emp2)
        else:
            result = classifier(emp1, emp2, cuda)
        pred_label = 0
        actual_label = row[2]
        if classifier is None:
            if result < threshold:
                pred_label = 1
        else:
            if result > threshold:
                pred_label = 1
        if pred_label != int(row[2]):
            error += 1
        confusion_matrix[int(not pred_label)][int(not actual_label)] += 1
        if pred_label == 1 and actual_label == 0:
            false_positive_rows.append(row)
        if pred_label == 0 and actual_label == 1:
            false_negative_rows.append(row)
        if actual_label == 1:
            avg_dist_same_persons += result
            total_dist_same_persons += 1
        if actual_label == 0:
            avg_dist_diff_persons += result
            total_dist_diff_persons += 1

        cnt += 1.0
        finished = int((cnt * 10) / data_cnt)

        remaining = 10 - finished
        te = time.time()
        time_sum += (te - ts)
        avg_time = time_sum / cnt
        time_remaing = avg_time * (data_cnt - cnt)
        accuracy = ((cnt - error) * 1.0 / cnt) * 100.0
        sys.stdout.write("\r Testing  [" + str(
            "=" * finished + str("." * remaining) + "] time remaining = " + str(
                time_remaing / 60.0)[:8]) + " Accuracy =" + str(round(accuracy, 3)))

    avg_dist_same_persons /= total_dist_same_persons
    avg_dist_diff_persons /= total_dist_diff_persons

    accuracy = ((cnt - error) * 1.0 / cnt) * 100.0
    print("Accuracy now equal --> {:.4f}%".format(accuracy))

    confusion_table = pd.DataFrame(confusion_matrix, index=['Predicted True', 'Predicted False'],
                                   columns=['Actual True', 'Actual False'])
    precision = (confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + 1)
    recall = (confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0] + 1)
    beta = 1.0
    beta_squared = beta ** 2
    fbeta_score = (1.0 + beta_squared) / ((beta_squared / recall) + (1 / precision))
    error_matrix = [['processed rows', int(cnt)],
                    ['Model accuracy on Proceed Faces %', round(accuracy, 2)],
                    ['False Positive', int(confusion_matrix[0][1])],
                    ['False Negative', int(confusion_matrix[1][0])],
                    ['precision', round(precision, 3)],
                    ['recall', round(recall, 3)],
                    ['fbeta-score', round(fbeta_score, 3)],
                    ['avg same person distance', round(avg_dist_same_persons, 3)],
                    ['avg diff person distance', round(avg_dist_diff_persons, 3)],
                    ['Model tolerance', threshold]
                    ]
    model_name = "convfacenet"

    error_table = pd.DataFrame(error_matrix, columns=['Mertic', 'Value'])

    if results_path is not None:
        file_name = f"{model_name}__err.csv"
        full_path = os.path.join(results_path, file_name)
        error_table.to_csv(full_path)

        file_name = f"{model_name}_conv.csv"
        full_path = os.path.join(results_path, file_name)
        confusion_table.to_csv(full_path)
    return error_table, confusion_table
