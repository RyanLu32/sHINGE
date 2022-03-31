import argparse, sys, json, pickle, torch, operator, random, os
from model import *
from itertools import product
import numpy as np
from random import randint
from random import randrange

from timeit import default_timer as timer
from datetime import timedelta
from random import shuffle

import torch.multiprocessing as mp

from copy import deepcopy
from pytictoc import TicToc
from collections import Counter
from batching import *

from multiprocessing import JoinableQueue, Queue, Process


def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus(model, n_rel_keys,
                                                                                        n_entities_values, whole_train,
                                                                                        whole_test, whole_valid,
                                                                                        list_of_testing_facts, device,
                                                                                        output_queue,
                                                                                        entityId2SparsifierType):
    with torch.no_grad():
        print("evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus on device cuda",
              device)
        model.to(device)
        model.eval()

        range_of_rel_keys = np.arange(n_rel_keys)
        range_of_entities_values = np.arange(n_entities_values)

        binary_hit1_keys = 0
        nary_hit1_keys = 0
        overall_hit1_keys = 0
        binary_hit3_keys = 0
        nary_hit3_keys = 0
        overall_hit3_keys = 0
        binary_hit10_keys = 0
        nary_hit10_keys = 0
        overall_hit10_keys = 0
        binary_mrr_keys = 0
        nary_mrr_keys = 0
        overall_mrr_keys = 0
        binary_mrr_values = 0
        nary_mrr_values = 0
        binary_hit1_values = 0
        nary_hit1_values = 0
        binary_hit3_values = 0
        nary_hit3_values = 0
        binary_hit10_values = 0
        nary_hit10_values = 0
        overall_hit1_values = 0
        overall_hit3_values = 0
        overall_hit10_values = 0
        overall_mrr_values = 0

        number_of_binary_facts_keys = 0
        number_of_nary_facts_keys = 0
        number_of_total_facts_keys = 0
        number_of_binary_facts_values = 0
        number_of_nary_facts_values = 0
        number_of_total_facts_values = 0

        head_value_mrr = 0
        head_value_hit10 = 0
        head_value_hit3 = 0
        head_value_hit1 = 0
        tail_value_mrr = 0
        tail_value_hit10 = 0
        tail_value_hit3 = 0
        tail_value_hit1 = 0
        number_of_head_value_facts = 0
        number_of_tail_value_facts = 0
        head_key_mrr = 0
        head_key_hit10 = 0
        head_key_hit3 = 0
        head_key_hit1 = 0
        tail_key_mrr = 0
        tail_key_hit10 = 0
        tail_key_hit3 = 0
        tail_key_hit1 = 0
        number_of_head_key_facts = 0
        number_of_tail_key_facts = 0

        binary_head_key_mrr = 0
        binary_head_key_hit10 = 0
        binary_head_key_hit3 = 0
        binary_head_key_hit1 = 0
        number_of_binary_head_key_facts = 0
        nary_head_key_mrr = 0
        nary_head_key_hit10 = 0
        nary_head_key_hit3 = 0
        nary_head_key_hit1 = 0
        number_of_nary_head_key_facts = 0
        binary_tail_key_mrr = 0
        binary_tail_key_hit10 = 0
        binary_tail_key_hit3 = 0
        binary_tail_key_hit1 = 0
        number_of_binary_tail_key_facts = 0
        nary_tail_key_mrr = 0
        nary_tail_key_hit10 = 0
        nary_tail_key_hit3 = 0
        nary_tail_key_hit1 = 0
        number_of_nary_tail_key_facts = 0

        binary_head_value_mrr = 0
        binary_head_value_hit10 = 0
        binary_head_value_hit3 = 0
        binary_head_value_hit1 = 0
        number_of_binary_head_value_facts = 0
        nary_head_value_mrr = 0
        nary_head_value_hit10 = 0
        nary_head_value_hit3 = 0
        nary_head_value_hit1 = 0
        number_of_nary_head_value_facts = 0
        binary_tail_value_mrr = 0
        binary_tail_value_hit10 = 0
        binary_tail_value_hit3 = 0
        binary_tail_value_hit1 = 0
        number_of_binary_tail_value_facts = 0
        nary_tail_value_mrr = 0
        nary_tail_value_hit10 = 0
        nary_tail_value_hit3 = 0
        nary_tail_value_hit1 = 0
        number_of_nary_tail_value_facts = 0

        keys_without_hrt_mrr = 0  # without hrt
        keys_without_hrt_hit10 = 0  # without hrt
        keys_without_hrt_hit3 = 0  # without hrt
        keys_without_hrt_hit1 = 0  # without hrt
        number_keys_without_hrt = 0  # without hrt
        values_without_hrt_mrr = 0  # without hrt
        values_without_hrt_hit10 = 0  # without hrt
        values_without_hrt_hit3 = 0  # without hrt
        values_without_hrt_hit1 = 0  # without hrt
        number_values_without_hrt = 0  # without hrt

        # t3 = TicToc()
        # t3.tic()
        for fact_progress, fact in enumerate(list_of_testing_facts):

            fact = list(fact)
            arity = int(len(fact) / 2)
            # num_process = 0
            # parse the fact by column and tile it
            for column in range(len(fact)):

                correct_index = fact[column]

                if column % 2 == 0:  # keys
                    if column == 0 or column == 2:
                        tiled_fact = np.array(fact * n_rel_keys).reshape(n_rel_keys, -1)
                        tiled_fact[:, 0] = range_of_rel_keys
                        tiled_fact[:, 2] = range_of_rel_keys
                    else:
                        tiled_fact = np.array(fact * n_rel_keys).reshape(n_rel_keys, -1)
                        tiled_fact[:, column] = range_of_rel_keys

                    if arity == 2:
                        number_of_binary_facts_keys = number_of_binary_facts_keys + 1
                    else:
                        number_of_nary_facts_keys = number_of_nary_facts_keys + 1

                    if arity == 2 and column == 0:
                        number_of_binary_head_key_facts += 1
                    elif arity == 2 and column == 2:
                        number_of_binary_tail_key_facts += 1
                    elif arity > 2 and column == 0:
                        number_of_nary_head_key_facts += 1
                    elif arity > 2 and column == 2:
                        number_of_nary_tail_key_facts += 1

                    if column > 3:
                        number_keys_without_hrt += 1

                else:
                    tiled_fact = np.array(fact * n_entities_values).reshape(n_entities_values, -1)
                    tiled_fact[:, column] = range_of_entities_values
                    if arity == 2:
                        number_of_binary_facts_values = number_of_binary_facts_values + 1
                    else:
                        number_of_nary_facts_values = number_of_nary_facts_values + 1

                    if arity == 2 and column == 1:
                        number_of_binary_head_value_facts += 1
                    elif arity > 2 and column == 1:
                        number_of_nary_head_value_facts += 1
                    elif arity == 2 and column == 3:
                        number_of_binary_tail_value_facts += 1
                    elif arity > 2 and column == 3:
                        number_of_nary_tail_value_facts += 1

                    if column > 3:
                        number_values_without_hrt += 1

                if column == 1:
                    number_of_head_value_facts += 1
                elif column == 3:
                    number_of_tail_value_facts += 1
                elif column == 0:
                    number_of_head_key_facts += 1
                elif column == 2:
                    number_of_tail_key_facts += 1

                tiled_fact = list(chunks(tiled_fact, 512))
                flag = 0
                for batch_it in range(len(tiled_fact)):

                    x_by_sparsifier, y_by_sparsifier = sort_new_batch_according_to_sparsifier_2(tiled_fact[batch_it],
                                                                                                None,
                                                                                                entityId2SparsifierType)

                    for j in x_by_sparsifier:
                        if batch_it == 0 and flag == 0:
                            num_tuple = len(x_by_sparsifier[j][0]) // arity
                            pred = model(np.array(x_by_sparsifier[j]), arity, num_tuple, device)
                            flag = 1
                        else:
                            num_tuple = len(x_by_sparsifier[j][0]) // arity
                            pred_tmp = model(np.array(x_by_sparsifier[j]), arity, num_tuple, device)
                            pred = torch.cat((pred, pred_tmp))

                sorted_pred = torch.argsort(pred, dim=0, descending=True)

                position_of_correct_fact_in_sorted_pred = 0
                for tmpxx in sorted_pred:
                    if tmpxx == correct_index:
                        break
                    tmp_list = deepcopy(fact)
                    tmp_list[column] = tmpxx.item()
                    tmpTriple = tuple(tmp_list)
                    if (len(whole_train) > arity - 2) and (tmpTriple in whole_train[arity - 2]):  # 2-ary in index 0
                        continue
                    elif (len(whole_valid) > arity - 2) and (tmpTriple in whole_valid[arity - 2]):  # 2-ary in index 0
                        continue
                    elif (len(whole_test) > arity - 2) and (tmpTriple in whole_test[arity - 2]):  # 2-ary in index 0
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred += 1

                if position_of_correct_fact_in_sorted_pred == 0:
                    if column % 2 == 0:  # keys
                        overall_hit1_keys = overall_hit1_keys + 1
                        if arity == 2:  # binary fact
                            binary_hit1_keys = binary_hit1_keys + 1
                        else:  # nary fact
                            nary_hit1_keys = nary_hit1_keys + 1
                        if column > 3:
                            keys_without_hrt_hit1 += 1  # without hrt
                    else:  # values
                        overall_hit1_values = overall_hit1_values + 1
                        if arity == 2:  # binary fact
                            binary_hit1_values = binary_hit1_values + 1
                        else:  # nary fact
                            nary_hit1_values = nary_hit1_values + 1
                        if column > 3:
                            values_without_hrt_hit1 += 1  # without hrt
                    if column == 1:
                        head_value_hit1 += 1
                        if arity == 2:
                            binary_head_value_hit1 += 1
                        else:
                            nary_head_value_hit1 += 1
                    elif column == 3:
                        tail_value_hit1 += 1
                        if arity == 2:
                            binary_tail_value_hit1 += 1
                        else:
                            nary_tail_value_hit1 += 1
                    elif column == 0:
                        head_key_hit1 += 1
                        if arity == 2:
                            binary_head_key_hit1 += 1
                        else:
                            nary_head_key_hit1 += 1
                    elif column == 2:
                        tail_key_hit1 += 1
                        if arity == 2:
                            binary_tail_key_hit1 += 1
                        else:
                            nary_tail_key_hit1 += 1

                if position_of_correct_fact_in_sorted_pred < 3:
                    if column % 2 == 0:  # keys
                        overall_hit3_keys = overall_hit3_keys + 1
                        if arity == 2:  # binary fact
                            binary_hit3_keys = binary_hit3_keys + 1
                        else:  # nary fact
                            nary_hit3_keys = nary_hit3_keys + 1
                        if column > 3:
                            keys_without_hrt_hit3 += 1  # without hrt
                    else:  # values
                        overall_hit3_values = overall_hit3_values + 1
                        if arity == 2:  # binary fact
                            binary_hit3_values = binary_hit3_values + 1
                        else:  # nary fact
                            nary_hit3_values = nary_hit3_values + 1
                        if column > 3:
                            values_without_hrt_hit3 += 1  # without hrt

                    if column == 1:
                        head_value_hit3 += 1
                        if arity == 2:
                            binary_head_value_hit3 += 1
                        else:
                            nary_head_value_hit3 += 1
                    elif column == 3:
                        tail_value_hit3 += 1
                        if arity == 2:
                            binary_tail_value_hit3 += 1
                        else:
                            nary_tail_value_hit3 += 1
                    elif column == 0:
                        head_key_hit3 += 1
                        if arity == 2:
                            binary_head_key_hit3 += 1
                        else:
                            nary_head_key_hit3 += 1
                    elif column == 2:
                        tail_key_hit3 += 1
                        if arity == 2:
                            binary_tail_key_hit3 += 1
                        else:
                            nary_tail_key_hit3 += 1

                if position_of_correct_fact_in_sorted_pred < 10:
                    if column % 2 == 0:  # keys
                        overall_hit10_keys = overall_hit10_keys + 1
                        if arity == 2:  # binary fact
                            binary_hit10_keys = binary_hit10_keys + 1
                        else:  # nary fact
                            nary_hit10_keys = nary_hit10_keys + 1
                        if column > 3:
                            keys_without_hrt_hit10 += 1  # without hrt
                    else:  # values
                        overall_hit10_values = overall_hit10_values + 1
                        if arity == 2:  # binary fact
                            binary_hit10_values = binary_hit10_values + 1
                        else:  # nary fact
                            nary_hit10_values = nary_hit10_values + 1
                        if column > 3:
                            values_without_hrt_hit10 += 1  # without hrt
                    if column == 1:
                        head_value_hit10 += 1
                        if arity == 2:
                            binary_head_value_hit10 += 1
                        else:
                            nary_head_value_hit10 += 1
                    elif column == 3:
                        tail_value_hit10 += 1
                        if arity == 2:
                            binary_tail_value_hit10 += 1
                        else:
                            nary_tail_value_hit10 += 1
                    elif column == 0:
                        head_key_hit10 += 1
                        if arity == 2:
                            binary_head_key_hit10 += 1
                        else:
                            nary_head_key_hit10 += 1
                    elif column == 2:
                        tail_key_hit10 += 1
                        if arity == 2:
                            binary_tail_key_hit10 += 1
                        else:
                            nary_tail_key_hit10 += 1

                if column % 2 == 0:  # keys
                    overall_mrr_keys = overall_mrr_keys + float(1 / (
                            position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if arity == 2:  # binary fact
                        binary_mrr_keys = binary_mrr_keys + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    else:  # nary fact
                        nary_mrr_keys = nary_mrr_keys + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if column > 3:  # without hrt
                        keys_without_hrt_mrr = keys_without_hrt_mrr + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                else:  # values
                    overall_mrr_values = overall_mrr_values + float(1 / (
                            position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if arity == 2:  # binary fact
                        binary_mrr_values = binary_mrr_values + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    else:  # nary fact
                        nary_mrr_values = nary_mrr_values + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                    if column > 3:  # without hrt
                        values_without_hrt_mrr = values_without_hrt_mrr + float(1 / (
                                position_of_correct_fact_in_sorted_pred + 1))  # +1 because otherwise if the predicted element is in top, it is going to divide by 0
                if column == 1:
                    head_value_mrr = head_value_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                    if arity == 2:
                        binary_head_value_mrr = binary_head_value_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                    else:
                        nary_head_value_mrr = nary_head_value_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                elif column == 3:
                    tail_value_mrr = tail_value_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                    if arity == 2:
                        binary_tail_value_mrr = binary_tail_value_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                    else:
                        nary_tail_value_mrr = nary_tail_value_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                elif column == 0:
                    head_key_mrr = head_key_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                    if arity == 2:
                        binary_head_key_mrr = binary_head_key_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                    else:
                        nary_head_key_mrr = nary_head_key_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                elif column == 2:
                    tail_key_mrr = tail_key_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                    if arity == 2:
                        binary_tail_key_mrr = binary_tail_key_mrr + float(
                            1 / (position_of_correct_fact_in_sorted_pred + 1))
                    else:
                        nary_tail_key_mrr = nary_tail_key_mrr + float(1 / (position_of_correct_fact_in_sorted_pred + 1))
            # num_process += 1
            # if num_process % 1 == 0:
            #     t3.toc()
            #     print("Process %d facts using %fS" % (num_process, t3.elapsed))
            #     t3.tic()
    output_message = {}

    number_of_total_facts_keys = number_of_binary_facts_keys + number_of_nary_facts_keys
    if number_of_total_facts_keys > 0:
        output_message["overall_keys"] = [device, number_of_total_facts_keys, overall_mrr_keys, overall_hit10_keys,
                                          overall_hit3_keys, overall_hit1_keys]

    number_of_total_facts_values = number_of_binary_facts_values + number_of_nary_facts_values
    if number_of_total_facts_values > 0:
        output_message["overall_values"] = [device, number_of_total_facts_values, overall_mrr_values,
                                            overall_hit10_values, overall_hit3_values, overall_hit1_values]

    if number_of_binary_facts_keys > 0:
        output_message["binary_keys"] = [device, number_of_binary_facts_keys, binary_mrr_keys, binary_hit10_keys,
                                         binary_hit3_keys, binary_hit1_keys]

    if number_of_binary_facts_values > 0:
        output_message["binary_values"] = [device, number_of_binary_facts_values, binary_mrr_values,
                                           binary_hit10_values, binary_hit3_values, binary_hit1_values]

    if number_of_nary_facts_keys > 0:
        output_message["nary_keys"] = [device, number_of_nary_facts_keys, nary_mrr_keys, nary_hit10_keys,
                                       nary_hit3_keys, nary_hit1_keys]

    if number_of_nary_facts_values > 0:
        output_message["nary_values"] = [device, number_of_nary_facts_values, nary_mrr_values, nary_hit10_values,
                                         nary_hit3_values, nary_hit1_values]

    if number_of_head_value_facts > 0:
        output_message["head_value_facts"] = [device, number_of_head_value_facts, head_value_mrr, head_value_hit10,
                                              head_value_hit3, head_value_hit1]

    if number_of_tail_value_facts > 0:
        output_message["tail_value_facts"] = [device, number_of_tail_value_facts, tail_value_mrr, tail_value_hit10,
                                              tail_value_hit3, tail_value_hit1]

    if number_of_head_key_facts > 0:
        output_message["head_key_facts"] = [device, number_of_head_key_facts, head_key_mrr, head_key_hit10,
                                            head_key_hit3, head_key_hit1]

    if number_of_tail_key_facts > 0:
        output_message["tail_key_facts"] = [device, number_of_tail_key_facts, tail_key_mrr, tail_key_hit10,
                                            tail_key_hit3, tail_key_hit1]

    if number_of_binary_head_key_facts > 0:
        output_message["binary_head_key_facts"] = [device, number_of_binary_head_key_facts, binary_head_key_mrr,
                                                   binary_head_key_hit10, binary_head_key_hit3, binary_head_key_hit1]

    if number_of_nary_head_key_facts > 0:
        output_message["nary_head_key_facts"] = [device, number_of_nary_head_key_facts, nary_head_key_mrr,
                                                 nary_head_key_hit10, nary_head_key_hit3, nary_head_key_hit1]

    if number_of_binary_tail_key_facts > 0:
        output_message["binary_tail_key_facts"] = [device, number_of_binary_tail_key_facts, binary_tail_key_mrr,
                                                   binary_tail_key_hit10, binary_tail_key_hit3, binary_tail_key_hit1]

    if number_of_nary_tail_key_facts > 0:
        output_message["nary_tail_key_facts"] = [device, number_of_nary_tail_key_facts, nary_tail_key_mrr,
                                                 nary_tail_key_hit10, nary_tail_key_hit3, nary_tail_key_hit1]

    if number_of_binary_head_value_facts > 0:
        output_message["binary_head_value_facts"] = [device, number_of_binary_head_value_facts, binary_head_value_mrr,
                                                     binary_head_value_hit10, binary_head_value_hit3,
                                                     binary_head_value_hit1]

    if number_of_nary_head_value_facts > 0:
        output_message["nary_head_value_facts"] = [device, number_of_nary_head_value_facts, nary_head_value_mrr,
                                                   nary_head_value_hit10, nary_head_value_hit3, nary_head_value_hit1]

    if number_of_binary_tail_value_facts > 0:
        output_message["binary_tail_value_facts"] = [device, number_of_binary_tail_value_facts, binary_tail_value_mrr,
                                                     binary_tail_value_hit10, binary_tail_value_hit3,
                                                     binary_tail_value_hit1]

    if number_of_nary_tail_value_facts > 0:
        output_message["nary_tail_value_facts"] = [device, number_of_nary_tail_value_facts, nary_tail_value_mrr,
                                                   nary_tail_value_hit10, nary_tail_value_hit3, nary_tail_value_hit1]

    if number_keys_without_hrt > 0:  # without hrt
        output_message["keys_without_hrt"] = [device, number_keys_without_hrt, keys_without_hrt_mrr,
                                              keys_without_hrt_hit10, keys_without_hrt_hit3, keys_without_hrt_hit1]

    if number_values_without_hrt > 0:  # without hrt
        output_message["values_without_hrt"] = [device, number_values_without_hrt, values_without_hrt_mrr,
                                                values_without_hrt_hit10, values_without_hrt_hit3,
                                                values_without_hrt_hit1]

    output_queue.put(output_message)

    return output_queue


def prepare_data_for_evaluation_and_evaluate_on_multiple_gpus(model, test, n_rel_keys, n_entities_values,
                                                              whole_train, whole_test, whole_valid, gpu_ids_splitted,
                                                              output_queue, entityId2SparsifierType):
    print("prepare_data_for_evaluation_and_evaluate_on_multiple_gpus")
    list_of_all_test_facts = []
    for test_fact_grouped_by_arity in test:
        for test_fact in test_fact_grouped_by_arity:
            list_of_all_test_facts.append(test_fact)

    shuffle(list_of_all_test_facts)
    slices = list(split_list(list_of_all_test_facts, len(gpu_ids_splitted)))

    jobs = []

    for slice_it, slice in enumerate(slices):
        device = gpu_ids_splitted[slice_it]
        print(device)
        current_job = mp.Process(
            target=evaluate_replicated_fact_with_correct_element_and_index_pre_stored_on_multiple_gpus, args=(
                model, n_rel_keys, n_entities_values, whole_train, whole_test, whole_valid, slices[slice_it],
                device,
                output_queue, entityId2SparsifierType))
        jobs.append(current_job)

    # start all job
    for current_job in jobs:
        current_job.start()

    # exit the completed processes
    for current_job in jobs:
        current_job.join()

    results = [output_queue.get() for current_job in jobs]
    weighted_scores = {}
    for dictionary in results:
        for task in dictionary:
            if task not in weighted_scores:
                weighted_scores[task] = []
                # dictionary[task][0] is device
                weighted_scores[task].append(dictionary[task][1])  # number of facts
                weighted_scores[task].append(dictionary[task][2])  # mrr
                weighted_scores[task].append(dictionary[task][3])  # hits@10
                weighted_scores[task].append(dictionary[task][4])  # hits@3
                weighted_scores[task].append(dictionary[task][5])  # hits@1
            else:
                weighted_scores[task][0] = weighted_scores[task][0] + dictionary[task][1]  # number of facts
                weighted_scores[task][1] = weighted_scores[task][1] + dictionary[task][2]  # mrr
                weighted_scores[task][2] = weighted_scores[task][2] + dictionary[task][3]  # hits@10
                weighted_scores[task][3] = weighted_scores[task][3] + dictionary[task][4]  # hits@3
                weighted_scores[task][4] = weighted_scores[task][4] + dictionary[task][5]  # hits@1

    overall_entity_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    overall_relation_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    triple_entity_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    triple_relation_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    hyperrelational_entity_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    hyperrelational_relation_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    hyperrelational_key_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}
    hyperrelational_value_prediction = {'mrr': 0, 'hits10': 0, 'hits1': 0}

    for task in weighted_scores:
        tot_facts = weighted_scores[task][0]
        mrr = weighted_scores[task][1]
        hits10 = weighted_scores[task][2]
        hits3 = weighted_scores[task][3]
        hits1 = weighted_scores[task][4]

        if task == "head_value_facts" or task == "tail_value_facts":  # table 2 head/tail prediction
            overall_entity_prediction['mrr'] += mrr / tot_facts
            overall_entity_prediction['hits10'] += hits10 / tot_facts
            overall_entity_prediction['hits1'] += hits1 / tot_facts

        if task == "head_key_facts" or task == "tail_key_facts":  # table 2 relation prediction
            overall_relation_prediction['mrr'] += mrr / tot_facts
            overall_relation_prediction['hits10'] += hits10 / tot_facts
            overall_relation_prediction['hits1'] += hits1 / tot_facts

        if task == "binary_head_value_facts" or task == "binary_tail_value_facts":  # table 4 triple fact head/tail prediction
            triple_entity_prediction['mrr'] += mrr / tot_facts
            triple_entity_prediction['hits10'] += hits10 / tot_facts
            triple_entity_prediction['hits1'] += hits1 / tot_facts

        if task == "binary_head_key_facts":  # table 4 triple fact relation prediction
            triple_relation_prediction['mrr'] += mrr / tot_facts
            triple_relation_prediction['hits10'] += hits10 / tot_facts
            triple_relation_prediction['hits1'] += hits1 / tot_facts

        if task == "nary_head_value_facts" or task == "nary_tail_value_facts":  # table 4 hyper relational fact head/tail prediction
            hyperrelational_entity_prediction['mrr'] += mrr / tot_facts
            hyperrelational_entity_prediction['hits10'] += hits10 / tot_facts
            hyperrelational_entity_prediction['hits1'] += hits1 / tot_facts

        if task == "nary_head_key_facts":  # table 4 hyper relational fact relation prediction
            hyperrelational_relation_prediction['mrr'] += mrr / tot_facts
            hyperrelational_relation_prediction['hits10'] += hits10 / tot_facts
            hyperrelational_relation_prediction['hits1'] += hits1 / tot_facts

        if task == "values_without_hrt":  # table 5 value prediction
            hyperrelational_value_prediction['mrr'] += mrr / tot_facts
            hyperrelational_value_prediction['hits10'] += hits10 / tot_facts
            hyperrelational_value_prediction['hits1'] += hits1 / tot_facts

        if task == "keys_without_hrt":  # table 5 key prediction
            hyperrelational_key_prediction['mrr'] += mrr / tot_facts
            hyperrelational_key_prediction['hits10'] += hits10 / tot_facts
            hyperrelational_key_prediction['hits1'] += hits1 / tot_facts

    overall_entity_prediction['mrr'] /= 2  # avg between head_value_facts and tail_value_facts
    overall_entity_prediction['hits10'] /= 2  # avg between head_value_facts and tail_value_facts
    overall_entity_prediction['hits1'] /= 2  # avg between head_value_facts and tail_value_facts

    overall_relation_prediction['mrr'] /= 2  # avg between head_key_facts and tail_key_facts
    overall_relation_prediction['hits10'] /= 2  # avg between head_key_facts and tail_key_facts
    overall_relation_prediction['hits1'] /= 2  # avg between head_key_facts and tail_key_facts

    triple_entity_prediction['mrr'] /= 2  # avg binary_head_value_facts and binary_tail_value_facts
    triple_entity_prediction['hits10'] /= 2  # avg binary_head_value_facts and binary_tail_value_facts
    triple_entity_prediction['hits1'] /= 2  # avg binary_head_value_facts and binary_tail_value_facts

    # NO need to divide triple_relation_prediction by 2

    hyperrelational_entity_prediction['mrr'] /= 2  # avg nary_head_value_facts and nary_tail_value_facts
    hyperrelational_entity_prediction['hits10'] /= 2  # avg nary_head_value_facts and nary_tail_value_facts
    hyperrelational_entity_prediction['hits1'] /= 2  # avg nary_head_value_facts and nary_tail_value_facts

    # NO need to divide hyperrelational_relation_prediction by 2

    # NO need to divide hyperrelational_value_prediction by 2

    # NO need to divide hyperrelational_key_prediction by 2

    print("Table2 head/tail prediction [mrr, hits@10, hits@1]:", "%.4f" % overall_entity_prediction['mrr'],
          "%.4f" % overall_entity_prediction['hits10'], "%.4f" % overall_entity_prediction['hits1'])

    print("Table2 relation prediction [mrr, hits@10, hits@1]:", "%.4f" % overall_relation_prediction['mrr'],
          "%.4f" % overall_relation_prediction['hits10'], "%.4f" % overall_relation_prediction['hits1'])

    print("Table4 triple fact head/tail prediction [mrr, hits@10, hits@1]:", "%.4f" % triple_entity_prediction['mrr'],
          "%.4f" % triple_entity_prediction['hits10'], "%.4f" % triple_entity_prediction['hits1'])

    print("Table4 triple fact relation prediction [mrr, hits@10, hits@1]:", "%.4f" % triple_relation_prediction['mrr'],
          "%.4f" % triple_relation_prediction['hits10'], "%.4f" % triple_relation_prediction['hits1'])

    print("Table4 hyper-relational fact head/tail prediction [mrr, hits@10, hits@1]:",
          "%.4f" % hyperrelational_entity_prediction['mrr'], "%.4f" % hyperrelational_entity_prediction['hits10'],
          "%.4f" % hyperrelational_entity_prediction['hits1'])

    print("Table4 hyper-relational fact relation prediction [mrr, hits@10, hits@1]:",
          "%.4f" % hyperrelational_relation_prediction['mrr'], "%.4f" % hyperrelational_relation_prediction['hits10'],
          "%.4f" % hyperrelational_relation_prediction['hits1'])

    print("Table5 hyper-relational fact value prediction [mrr, hits@10, hits@1]:",
          "%.4f" % hyperrelational_value_prediction['mrr'], "%.4f" % hyperrelational_value_prediction['hits10'],
          "%.4f" % hyperrelational_value_prediction['hits1'])

    print("Table5 hyper-relational fact key prediction [mrr, hits@10, hits@1]:",
          "%.4f" % hyperrelational_key_prediction['mrr'], "%.4f" % hyperrelational_key_prediction['hits10'],
          "%.4f" % hyperrelational_key_prediction['hits1'])

    ### END OF PARALLELIZATION ###
    print("Evaluation is over.")


def build_entity2types_dictionaries(dataset_name, entities_values2id):
    entityName2entityTypes = {}
    entityId2entityTypes = {}
    entityType2entityNames = {}
    entityType2entityIds = {}

    entity2type_file = open(dataset_name, "r")

    for line in entity2type_file:
        splitted_line = line.strip().split("\t")
        entity_name = splitted_line[0][8:]
        entity_type = splitted_line[1][6:]

        if entity_name not in entityName2entityTypes:
            entityName2entityTypes[entity_name] = []
        if entity_type not in entityName2entityTypes[entity_name]:
            entityName2entityTypes[entity_name].append(entity_type)

        if entity_type not in entityType2entityNames:
            entityType2entityNames[entity_type] = []
        if entity_name not in entityType2entityNames[entity_type]:
            entityType2entityNames[entity_type].append(entity_name)

        entity_id = entities_values2id[entity_name]
        if entity_id not in entityId2entityTypes:
            entityId2entityTypes[entity_id] = []
        if entity_type not in entityId2entityTypes[entity_id]:
            entityId2entityTypes[entity_id].append(entity_type)

        if entity_type not in entityType2entityIds:
            entityType2entityIds[entity_type] = []
        if entity_id not in entityType2entityIds[entity_type]:
            entityType2entityIds[entity_type].append(entity_id)

    entity2type_file.close()

    return entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds


def build_type2id_v2(inputData):
    type2id = {}
    id2type = {}
    type_counter = 0
    with open(inputData) as entity2type_file:
        for line in entity2type_file:
            splitted_line = line.strip().split("\t")
            entity_type = splitted_line[1][6:]

            if entity_type not in type2id:
                type2id[entity_type] = type_counter
                id2type[type_counter] = entity_type
                type_counter += 1

    entity2type_file.close()
    return type2id, id2type


def build_typeId2frequency(dataset_name, type2id):
    if "type2relation2type_ttv" in dataset_name:
        typeId2frequency = {}
        type_relation_type_file = open(dataset_name, "r")

        for line in type_relation_type_file:
            splitted_line = line.strip().split("\t")
            head_type = splitted_line[0][6:]
            tail_type = splitted_line[2][6:]
            head_type_id = type2id[head_type]
            tail_type_id = type2id[tail_type]

            if head_type_id not in typeId2frequency:
                typeId2frequency[head_type_id] = 0
            if tail_type_id not in typeId2frequency:
                typeId2frequency[tail_type_id] = 0

            typeId2frequency[head_type_id] += 1
            typeId2frequency[tail_type_id] += 1

        type_relation_type_file.close()
    elif "type2relation2type2key2type_ttv" in dataset_name:
        typeId2frequency = {}
        type_relation_type_file = open(dataset_name, "r")
        for line in type_relation_type_file:
            splitted_line = line.strip().split("\t")
            for i in range(0, len(splitted_line), 2):
                value_type = splitted_line[i][6:]
                value_type_id = type2id[value_type]
                if value_type_id not in typeId2frequency:
                    typeId2frequency[value_type_id] = 0
                typeId2frequency[value_type_id] += 1
        type_relation_type_file.close()
    return typeId2frequency


def build_entityId2SparsifierType(entities_values2id, type2id, entityId2entityTypes, sparsifier, typeId2frequency,
                                  unk_entity_type_id):
    entityId2SparsifierType = {}
    if sparsifier > 0:
        for i in entities_values2id:
            entityId = entities_values2id[i]
            if entityId in entityId2entityTypes:
                entityTypes = entityId2entityTypes[entityId]
                entityTypeIds = []
                for j in entityTypes:
                    entityTypeIds.append(type2id[j])

                current_freq = {}
                for typeId in entityTypeIds:
                    current_freq[typeId] = typeId2frequency[typeId]

                sorted_current_freq = sorted(current_freq.items(), key=lambda kv: kv[1],
                                             reverse=True)[:sparsifier]
                topNvalueTypes = [item[0] for item in sorted_current_freq]
                entityId2SparsifierType[entityId] = topNvalueTypes

            else:
                entityId2SparsifierType[entityId] = [unk_entity_type_id]
        return entityId2SparsifierType
    else:
        print("SPARSIFIER ERROR!")


def sort_new_batch_according_to_sparsifier_2(x_batch, y_batch, entityId2SparsifierType):
    x_by_sparsifier = {}
    y_by_sparsifier = {}
    for item in range(len(x_batch)):
        value_list = x_batch[item][1::2]
        type_list = [entityId2SparsifierType[i] for i in value_list]
        item_plus = list(x_batch[item])
        for i in product(*type_list):
            item_plus.extend(i)
        if len(item_plus) not in x_by_sparsifier:
            x_by_sparsifier[len(item_plus)] = []
            x_by_sparsifier[len(item_plus)].append(item_plus)
            if not (y_batch is None):
                y_by_sparsifier[len(item_plus)] = []
                y_by_sparsifier[len(item_plus)].append(y_batch[item])
        else:
            if item_plus not in x_by_sparsifier[len(item_plus)]:
                x_by_sparsifier[len(item_plus)].append(item_plus)
                if not (y_batch is None):
                    y_by_sparsifier[len(item_plus)].append(y_batch[item])

    return x_by_sparsifier, y_by_sparsifier



def main():
    # parse input arguments
    parser = argparse.ArgumentParser(description="Model's hyperparameters")
    parser.add_argument('--indir', type=str, help='Input dir of train, test and valid data')
    parser.add_argument('--epochs', default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--num_filters', type=int, default=200, help='number of filters CNN')
    parser.add_argument('--embsize', default=100, help='Embedding size (default: 100)')
    parser.add_argument('--learningrate', default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--outdir', type=str, help='Output dir of model')
    parser.add_argument('--load', default='False',
                        help='If true, it loads a saved model in dir outdir and evaluate it (default: False)')
    parser.add_argument('--gpu_ids', default='0,1,2,3',
                        help='Comma-separated gpu id used to paralellize the evaluation (default: 0,1,2,3)')
    parser.add_argument('--num_negative_samples', type=int, default=1,
                        help='number of negative samples for each positive sample')
    parser.add_argument('--sparsifier', type=int, default=-1,
                        help='if type frequency is less than K in ranking, set its entry to 0 in the img. If its value is <=0 then it will not sparsify the matrix')
    args = parser.parse_args()
    print("\n\n************************")
    for e in vars(args):
        print(e, getattr(args, e))
    print("************************\n\n")

    gpu_ids_splitted = list(map(int, args.gpu_ids.split(",")))

    if args.load == 'True':
        t2 = TicToc()
        print("Loading and evaluating model in", args.outdir)
        mp.set_start_method('spawn')
        with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
            data_info = pickle.load(fin)
        test = data_info["test_facts"]
        rel_keys2id = data_info['roles_indexes']  # keys_indexes
        entities_values2id = data_info['values_indexes']  # values_indexes
        n_rel_keys = len(rel_keys2id)
        n_entities_values = len(entities_values2id)

        with open(args.indir + "/dictionaries_and_facts_permutate.bin", 'rb') as fin:
            data_info1 = pickle.load(fin)
        whole_train = data_info1["train_facts"]
        whole_valid = data_info1["valid_facts"]
        whole_test = data_info1['test_facts']

        entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds = \
            build_entity2types_dictionaries(args.indir + '/entity2types_ttv.txt', entities_values2id)

        ## img matrix
        type2id, id2type = build_type2id_v2(args.indir + '/entity2types_ttv.txt')

        entity_typeId2frequency = build_typeId2frequency(args.indir + '/type2relation2type_ttv.txt', type2id)
        value_typeId2frequency = build_typeId2frequency(args.indir + '/type2relation2type2key2type_ttv.txt', type2id)

        entity_typeId2frequency, value_typeId2frequency = Counter(entity_typeId2frequency), Counter(
            value_typeId2frequency)
        typeId2frequency = dict(entity_typeId2frequency + value_typeId2frequency)

        unk_entity_type_id = len(type2id)

        entityId2SparsifierType = build_entityId2SparsifierType(entities_values2id, type2id, entityId2entityTypes,
                                                                args.sparsifier,
                                                                typeId2frequency,
                                                                unk_entity_type_id)

        model = torch.load(args.outdir)

        t2.tic()
        output_queue = mp.Queue()
        prepare_data_for_evaluation_and_evaluate_on_multiple_gpus(model, test, n_rel_keys, n_entities_values,
                                                                  whole_train, whole_test, whole_valid,
                                                                  gpu_ids_splitted, output_queue,
                                                                  entityId2SparsifierType)
        t2.toc()
        print("Evaluation running time (seconds):", t2.elapsed)

        print("END OF SCRIPT!")

        sys.stdout.flush()

    else:

        # Load training data
        with open(args.indir + "/dictionaries_and_facts.bin", 'rb') as fin:
            data_info = pickle.load(fin)
        train = data_info["train_facts"]
        valid = data_info["valid_facts"]
        test = data_info['test_facts']
        entities_values2id = data_info['values_indexes']  # values_indexes
        rel_keys2id = data_info['roles_indexes']  # keys_indexes
        key_val = data_info['role_val']

        n_entities_values = len(entities_values2id)
        n_rel_keys = len(rel_keys2id)
        print("Unique number of relations and keys:", n_rel_keys)
        print("Unique number of entities and values:", n_entities_values)

        # Load the whole dataset for negative sampling in "batching.py"
        with open(args.indir + "/dictionaries_and_facts_permutate.bin", 'rb') as fin:
            data_info1 = pickle.load(fin)
        whole_train = data_info1["train_facts"]
        whole_valid = data_info1["valid_facts"]
        whole_test = data_info1['test_facts']

        # entity2type/value2type
        # attention: some entities and values in entities_values2id may not have types

        entityName2entityTypes, entityId2entityTypes, entityType2entityNames, entityType2entityIds = \
            build_entity2types_dictionaries(args.indir + '/entity2types_ttv.txt', entities_values2id)

        ## img matrix
        type2id, id2type = build_type2id_v2(args.indir + '/entity2types_ttv.txt')

        entity_typeId2frequency = build_typeId2frequency(args.indir + '/type2relation2type_ttv.txt', type2id)
        value_typeId2frequency = build_typeId2frequency(args.indir + '/type2relation2type2key2type_ttv.txt', type2id)

        entity_typeId2frequency_tmp, value_typeId2frequency_tmp = Counter(entity_typeId2frequency), Counter(
            value_typeId2frequency)
        typeId2frequency = dict(entity_typeId2frequency_tmp + value_typeId2frequency_tmp)

        unk_entity_type_id = len(type2id)

        entityId2SparsifierType = build_entityId2SparsifierType(entities_values2id, type2id, entityId2entityTypes,
                                                                args.sparsifier,
                                                                typeId2frequency,
                                                                unk_entity_type_id)

        # Prepare validation and test facts
        x_valid = []
        y_valid = []
        for k in valid:
            x_valid.append(np.array(list(k.keys())).astype(
                np.int32))  # x_valid[0] = [[roleid1,valueid1,roleid2,valueid2],[],...]  s.t. roleid1 = roleid2
            y_valid.append(np.array(list(k.values())).astype(np.float32))
        x_test = []
        y_test = []
        for k in test:
            x_test.append(np.array(list(k.keys())).astype(np.int32))
            y_test.append(np.array(list(k.values())).astype(np.int32))

        model = HINGE(len(rel_keys2id), len(entities_values2id), len(type2id) + 1, int(args.embsize),
                      int(args.num_filters))

        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())

        opt = torch.optim.Adam(model.parameters(), lr=float(
            args.learningrate))  # parameters returns all tensors that represents the parameters of the model
        mp.set_start_method('spawn')  # parallel processing
        t1 = TicToc()  # timer t1
        t2 = TicToc()  # timer t2

        n_batches_per_epoch = []
        for i in train:
            ll = len(i)
            if ll == 0:
                n_batches_per_epoch.append(0)
            else:
                n_batches_per_epoch.append(int((ll - 1) / args.batchsize) + 1)  # each epoch contains xx batches

        epoch = 0
        device = torch.device("cuda")

        for epoch in range(1, int(args.epochs) + 1):  # per epoch per arity per batch
            t1.tic()
            model.train()
            model.to(device)
            train_loss = 0

            for i in range(len(train)):  # batch_number == i
                train_i_indexes = np.array(list(train[i].keys())).astype(np.int32)
                train_i_values = np.array(list(train[i].values())).astype(np.float32)

                for batch_num in range(n_batches_per_epoch[i]):
                    arity = i + 2

                    x_batch, y_batch = Batch_Loader(train_i_indexes, train_i_values, n_entities_values, n_rel_keys,
                                                    key_val, args.batchsize, arity, whole_train[i],
                                                    args.num_negative_samples)

                    x_by_sparsifier, y_by_sparsifier = sort_new_batch_according_to_sparsifier_2(x_batch, y_batch,
                                                                                                entityId2SparsifierType)

                    pred_final = torch.FloatTensor().cuda()
                    for j in x_by_sparsifier:
                        num_tuple = len(x_by_sparsifier[j][0]) // arity
                        pred = model(np.array(x_by_sparsifier[j]), arity, num_tuple, device)
                        pred_tmp = pred * torch.FloatTensor(np.array(y_by_sparsifier[j])).cuda() * (-1)
                        pred_final = torch.cat((pred_final, pred_tmp), 0)

                    loss = model.loss(pred_final).mean()  # Softplus
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
            t1.toc()
            print("End of epoch", epoch, "- train_loss:", train_loss, "- training time (seconds):", t1.elapsed)

            sys.stdout.flush()

        print("END OF EPOCHS")

        # SAVE THE LAST MODEL
        file_name = "HINGE_type_" + str(
            args.batchsize) + "_" + args.epochs + "_" + str(args.num_filters) + "_" + args.embsize + "_" + args.learningrate + "_" + str(
            args.sparsifier)
        print("Saving the model trained at epoch", epoch, "in:", args.outdir + '/' + file_name)
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        torch.save(model, args.outdir + '/' + file_name)
        print("Model saved")

        t2.tic()
        output_queue = mp.Queue()

        prepare_data_for_evaluation_and_evaluate_on_multiple_gpus(model, test, epoch, n_rel_keys, n_entities_values,
                                                                  whole_train, whole_test, whole_valid,
                                                                  gpu_ids_splitted, output_queue,
                                                                  entityId2SparsifierType)
        t2.toc()
        print("Evaluation last epoch ", epoch, "- running time (seconds):", t2.elapsed)

        print("END OF SCRIPT!")

        sys.stdout.flush()


if __name__ == '__main__':
    main()
