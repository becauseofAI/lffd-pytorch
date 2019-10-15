# -*- coding: utf-8 -*-

import numpy
import torch


class Metric:
    def __init__(self, num_scales):
        self.num_scales = num_scales
        self.sum_metric = [0.0 for i in range(num_scales * 2)]
        self.num_update = 0
        self.multiply_factor = 10000

    def update(self, loss_branch):
        for i in range(self.num_scales):
            loss_score = loss_branch[i * 2]
            loss_bbox = loss_branch[i * 2 + 1]

            self.sum_metric[i * 2] += loss_score
            self.sum_metric[i * 2 + 1] += loss_bbox

        self.num_update += 1

    def get(self):
        return_string_list = []
        for i in range(self.num_scales):
            return_string_list.append('cls_loss_score_' + str(i))
            return_string_list.append('reg_loss_bbox_' + str(i))

        return return_string_list, [m / self.num_update * self.multiply_factor for i, m in enumerate(self.sum_metric)]

    def reset(self):
        self.sum_metric = [0.0 for i in range(self.num_scales * 2)]
        self.num_update = 0
