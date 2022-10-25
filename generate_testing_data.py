from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


# generate_training_data

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples = len(df[0])  # 720
    num_nodes = len(df)  # 630

    data1 = np.expand_dims(df.T, axis=-1)
    data_list = [data1]

    data = np.concatenate(data_list, axis=-1)
    # print(data.shape) # (720, 630, 1)
    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y


def generate_train_val_test(output_dir, data_raw):
    
    datatmp = []
    final = []
    for i in range(len(data_raw)):
      datatmp = data_raw[i][1]
      if (len(datatmp)) != 2880:
            datatmp += [datatmp[-1]] * (2880-len(datatmp))

      final.append(datatmp)
    # cut the dataset if shorter time series is needed. for example: 5 min.
    #final_cut = final[time1:time1+30]
    df = np.array(final)


    print('df',df.shape)

    # 0 is the latest observed sample.
    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(-11, 1, 1),))
    )
    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, 13, 1))
    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=False,
        add_day_in_week=False,
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    num_samples = x.shape[0]
    num_test = round(num_samples * 1)

    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, "%s.npz" % cat),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )



def main(output_dir, data_raw):
    print("Generating testing data")
    data_raw = data_raw
    generate_train_val_test(output_dir, data_raw)


if __name__ == "__main__":
    output_dir = "datasets/10s_8h_2days_test/"
    data_raw = np.load('rawdata/npy/8h_2days_fill.npy', allow_pickle=True)

    main(output_dir, data_raw)
