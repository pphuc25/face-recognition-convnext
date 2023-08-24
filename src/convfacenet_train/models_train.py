import os
import sys
import time
from datetime import datetime

import pandas as pd
from torch import optim, no_grad
from torch.nn import TripletMarginLoss, BCELoss
import torch


def model_test(model, test_loader, loss_function, test_mod, cuda):
    batch_size = test_loader.batch_size
    no_batches = len(test_loader)
    dataset_size = float(len(test_loader.dataset))
    model.eval()
    if cuda:
        model.cuda()
    loss_sum = 0.0
    cnt = 0.0
    time_sum = 0.0
    with no_grad():
        for data_row in test_loader:
            ts = time.time()

            if test_mod == "triplet":
                anchor_img, positive_img, negative_img = data_row
                loss_sum += triplet_test_step(model, loss_function, anchor_img, positive_img, negative_img,
                                              cuda) * batch_size
            elif test_mod == "pair":
                face_x, face_y, label = data_row
                loss_sum += pair_test_step(model, loss_function, face_x, face_y, label, cuda) * batch_size

            else:
                raise ValueError("invalid test mod")
            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            te = time.time()
            time_sum += (te - ts)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)

            sys.stdout.write("\r Testing  [" + str(
                "=" * finished + str("." * remaining) + "] time remaining (m) = " + str(
                    time_remaing / 60.0)[:8] + " Avg Test_Loss=" + str(loss_sum / (cnt * batch_size))[:8]))
            sys.stdout.flush()
            test_loss = loss_sum / dataset_size

    sys.stdout.write("\r Testing  [" + str(
        "=" * 10 + "] time taken (m) = " + str(
            time_sum / 60.0)[:8] + " Avg Test_Loss=" + str(loss_sum / (cnt * batch_size))[:8]))
    sys.stdout.flush()

    return test_loss


def model_train(model, epochs, learn_rate, train_loader, test_loader, train_mod, cuda=False, weight_saving_path=None,
                epoch_data_saving_path=None, notes=None, **kwargs
                ):
    if "optimizer" not in kwargs:
        optimizer = optim.Adam(model.parameters(), lr=learn_rate, weight_decay=1e-4)
    else:
        optimizer = kwargs["optimizer"]

    if "criterion" in kwargs:
        loss_function = kwargs["criterion"]
    else:
        if train_mod == "triplet":
            loss_function = TripletMarginLoss()
        elif train_mod == "pair":
            loss_function = BCELoss()
        else:
            raise ValueError("invalid train mod")

    batch_size = train_loader.batch_size
    no_batches = len(train_loader)
    dataset_size = float(len(train_loader.dataset))

    if cuda:
        model.cuda()
    train_losses = []
    test_losses = []
    print("Testing before training ...")
    min_test_loss = model_test(model, test_loader, loss_function, train_mod, cuda)
    print(f"Test Loss before Training={min_test_loss}")
    print("-----------------------------------------------------")
    for e in range(epochs):
        model.train()
        epoch_start_time = time.time()
        loss_sum = 0.0
        model.train()
        cnt = 0.0
        time_sum = 0.0
        for data in train_loader:
            batch_start_t = time.time()

            if train_mod == "triplet":
                anchor_img, positive_img, negative_img = data
                loss_sum += triplet_train_step(model, optimizer, loss_function, anchor_img, positive_img, negative_img,
                                               cuda) * batch_size
            elif train_mod == "pair":
                face_x, face_y, label = data
                loss_sum += pair_train_step(model, optimizer, loss_function, face_x, face_y, label, cuda) * batch_size
            else:
                raise ValueError("invalid train mod")

            cnt += 1.0
            finished = int((cnt * 10) / no_batches)
            remaining = 10 - finished
            batch_end_time = time.time()
            time_sum += (batch_end_time - batch_start_t)
            avg_time = time_sum / cnt
            time_remaing = avg_time * (no_batches - cnt)
            sys.stdout.flush()
            sys.stdout.write("\r epoch " + str(e + 1) + " [" + str(
                "=" * int((cnt * 10) / no_batches) + str("." * remaining) + "] time remaining (m) = " + str(
                    time_remaing / 60.0)[:8]) + " Avg Train_Loss=" + str(loss_sum / (cnt * batch_size))[:8])
        sys.stdout.write("\r epoch " + str(e + 1) + " [" + str(
            "=" * 10 + "] time Taken (git m) = " + str(
                time_sum / 60.0)[:8]) + " Avg Train_Loss=" + str(loss_sum / (cnt * batch_size))[:8])
        sys.stdout.flush()

        train_loader.dataset.print_usage_statistics()

        train_loss = loss_sum / dataset_size
        train_losses.append(train_loss)

        test_loss = model_test(model, test_loader, loss_function, train_mod, cuda)
        test_losses.append(test_loss)
        epoch_end_time = time.time()

        print(f" epoch {e + 1} train_loss ={train_loss} test_loss={test_loss}")
        if test_loss < min_test_loss:
            print(
                f"new minimum test loss {str(test_loss)[:8]} ", end=" ")
            if weight_saving_path is not None:
                save_train_weights(model, train_loss, test_loss, weight_saving_path)
                print("achieved, model weights saved", end=" ")
            print()

            min_test_loss = test_loss

        if train_loss < test_loss:
            print("!!!Warning Overfitting!!!")
        epoch_time_taken = round((epoch_end_time - epoch_start_time) / 60, 1)
        save_epochs_to_csv(epoch_data_saving_path, train_loss, len(train_loader.dataset), test_loss,
                           len(test_loader.dataset), epoch_time_taken, notes)
        print("-----------------------------------------------------")

    return train_losses, test_losses


def save_train_weights(model, train_loss, test_loss, saving_path):
    """
    saves model weights with file name format Day_Month Hour_minute train_(train_loss) test_(test_loss)
    :param model: model object
    :param train_loss: train loss (float)
    :param test_loss: test loss (float)
    :param saving_path: the path you want to save the weights in
    :return: the full path of the saved file (saving_path+filename)
    """
    weight_file_name = f"{datetime.now().strftime('%m_%d %H_%M')} Train_({str(train_loss)[:8]}) Test_({str(test_loss)[:8]}).pt"
    full_path = f"{saving_path}/{weight_file_name}"

    torch.save(model.state_dict(), full_path)
    return full_path


def save_epochs_to_csv(csv_save_path, train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes=None):
    if notes is None:
        notes = ""
    date_now = datetime.now()
    if len(csv_save_path) == 0:
        full_path = "train_data.csv"
    else:
        full_path = f"{csv_save_path}/train_data.csv"
    row = [[train_loss, no_train_rows, test_loss, no_test_rows, time_taken, notes, date_now.strftime('%d/%m/%Y'),
            date_now.strftime('%H:%M:00')]]
    df = pd.DataFrame(row,
                      columns=["Train Loss", "no train rows", "Test Loss", "No test rows", "Time taken (M)", "Notes",
                               "Date", "Time"])

    if not os.path.exists(full_path):
        df.to_csv(full_path, index=False)
    else:
        df.to_csv(full_path, mode='a', header=False, index=False)


def triplet_train_step(model, optimizer, loss_function, anchor_img, positive_img, negative_img, cuda):
    optimizer.zero_grad()
    anchor_img.requires_grad = True
    positive_img.requires_grad = True
    negative_img.requires_grad = True

    if cuda:
        anchor_img, positive_img, negative_img = anchor_img.cuda(), positive_img.cuda(), negative_img.cuda()

    anchor_vector = model(anchor_img)
    positive_vector = model(positive_img)
    negative_vector = model(negative_img)

    loss = loss_function(anchor_vector, positive_vector, negative_vector)
    loss.backward()

    optimizer.step()
    return loss.item()


def triplet_test_step(model, loss_function, anchor_img, positive_img, negative_img, cuda):
    if cuda:
        anchor_img, positive_img, negative_img = anchor_img.cuda(), positive_img.cuda(), negative_img.cuda()
    anchor_vector = model(anchor_img)
    positive_vector = model(positive_img)
    negative_vector = model(negative_img)
    loss = loss_function(anchor_vector, positive_vector, negative_vector)
    return loss.item()


def pair_train_step(model, optimizer, loss_function, face_x, face_y, label, cuda):
    optimizer.zero_grad()
    if cuda:
        face_x, face_y = face_x.cuda(), face_y.cuda()
        label = label.cuda()
    predicted_result = model(face_x, face_y)

    loss = loss_function(predicted_result, label)
    loss.backward()
    optimizer.step()
    return float(loss)


def pair_test_step(model, loss_function, face_x, face_y, label, cuda):
    if cuda:
        face_x, face_y = face_x.cuda(), face_y.cuda()
        label = label.cuda()
    predicted_result = model(face_x, face_y)
    loss = loss_function(predicted_result, label)
    return float(loss)

