import argparse
import itertools
import operator
import numpy as np
import os
from fuzzy_fusion import fuzzy_fusion

parser = argparse.ArgumentParser(description='PyTorch Three-Stream Action Recognition - Fusion')
parser.add_argument('npy_dir', type=str, help='path to npy files')
parser.add_argument('val_dir', type=str, help='path to npy files')
parser.add_argument('-s', '--split', default=-1, type=int, metavar='S',
          help='which split of data to work on (default: -1 [all])')
parser.add_argument('-a', default="inception_v3", type=str,
                        help='architecture (default: inception_v3): inception_v3 | resnet152',
                        choices=["inception_v3","resnet152"])
parser.add_argument('-d', default="ucf101", type=str,
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])
parser.add_argument('-m', default=["rgb","flow","rhythm"], type=str, nargs='+',
                        help='modalities (default: rgb,flow,rhythm)')
parser.add_argument('-w', default=[2.,3.,1.], type=float, nargs='+',
                        help='modalities weights (default: 2.,3.,1.)')

parser.add_argument('-f', action='store_true',
                    help='use fuzzy fusion')
parser.add_argument('-g', action='store_true',
                    help='use grid search')
parser.add_argument('-n', action='store_true',
                    help='softmax norm')

def softmax(x):
    x_ = np.exp(x)
    x_ = np.array([ i/sum(i) for i in x_ ])
    return x_
   
    """Compute softmax values for each sets of scores in x."""
    #return np.exp(x) / np.sum(np.exp(x), axis=1)


def obtain_accuracy(data_complete, ground_truth, text='rgb'):
    """
    This function calculates the value of the acurracy of a
    certain test(rgb, visual rhythm or optical flow)
    """
    match_count = 0
    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        if ground_truth[i]==index:
            match_count += 1
    text = "Accuracy for "+text+" : %4.4f"
    acurracy = float(match_count)/len(data_complete)
    return text, acurracy


def obtain_ground_truth(path_file):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []
    class_name = [None]*101
    for line in lines_file:
        line_info = line.split(' ')
        line_name = line.split('_')
        ground_truth.append(int(line_info[2]))
        class_name[int(line_info[2])] = line_name[1]

    class_name = np.array(class_name)
    np.array(ground_truth)
    return class_name, np.array(ground_truth)


def partial_result(npy_fmt, val_fmt, modalities, parameters, split, use_fuzzy=False, softmax_norm=False):
    path_file = val_fmt % str(split)
    class_name, ground_truth = obtain_ground_truth(path_file)
    acc_list = []

    # data = np.load(npy_fmt % (modalities[0], str(split)))
    # text, accuracy = obtain_accuracy(data, ground_truth, modalities[0])
    # print(text % (accuracy))
    # acc_list.append(accuracy)
    new_data = 0

    all_data = []
    for i in range(0, len(modalities)):
        data = np.load(npy_fmt % (modalities[i], str(split)))
        if softmax_norm: data = softmax(data)
        text, accuracy = obtain_accuracy(data, ground_truth, modalities[i])
        print(text % (accuracy))
        acc_list.append(accuracy)
        new_data = new_data + parameters[i]*data
        all_data.append(data)

    if use_fuzzy:
        new_data = fuzzy_fusion(all_data, parameters)

    text, accuracy = obtain_accuracy(new_data, ground_truth, "+".join(modalities))
    text = text % accuracy
    acc_list.append(accuracy)
    print("Parameters : ", parameters, text)

    return acc_list

def gs_result(npy_fmt, val_fmt, modalities, split,softmax_norm=False):
    path_file = val_fmt % str(split)
    class_name, ground_truth = obtain_ground_truth(path_file)

    all_data = []
    for i in range(0, len(modalities)):
        data = np.load(npy_fmt % (modalities[i], str(split)))
        if softmax_norm: data = softmax(data)
        all_data.append(data)

    weights = itertools.product(range(0,100,5),repeat=len(modalities))
    max_acc = 0.
    max_weight = [0]*len(modalities)
    for tup in weights:
        parameters = [x/10. for x in tup]
        #new_data = itertools.dotproduct(parameters, all_data)
        new_data = sum(map(operator.mul, parameters, all_data))
        text, accuracy = obtain_accuracy(new_data, ground_truth, "+".join(modalities))
        if accuracy > max_acc:
            max_acc, max_weight = accuracy, parameters

    print ("Split ", split, "Max Acc: ",  max_acc, "Parameters: ", max_weight)

def main():
    args = parser.parse_args()

    # dataset, modality, architecture, split
    npy_fmt = os.path.join(args.npy_dir, "%s_%s_%s_s%s.npy") % (args.d, "%s",args.a,"%s")
    val_fmt = os.path.join(args.val_dir, "%s/val_split%s.txt" % (args.d, "%s"))

    w = args.w
    # for w1 in np.arange(1, 11, 1):
    #     for w2 in np.arange(1, 11, 1):
    #         #for w3 in np.arange(1, 11, 1):
    #         w = [w1, w2]
    if (args.split != -1):
        acc_list = partial_result(npy_fmt, val_fmt, args.m,
                                  w, args.split, args.f, args.n)
    else:
        if(args.g):
            for s in range(1,4):
                gs_result(npy_fmt, val_fmt, args.m, s, args.n)
        else:
            acc_list = np.zeros(len(args.m)+1)
            for s in range(1, 4):
                acc_list += np.array(partial_result(npy_fmt, val_fmt, args.m,
                                                w, s, args.f, args.n))
            acc_list /= 3.

    print("Average")
    for i in range(len(args.m)):
        print(args.m[i], "%.4f" % acc_list[i])
    print("fusion", "%.4f" % acc_list[-1])


if __name__ == '__main__':
    main()    
