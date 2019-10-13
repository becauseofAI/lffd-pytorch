# -*- coding: utf-8 -*-

import numpy
import torch


class Metric:

    def __init__(self, num_scales):
        self.sum_metric = [0.0 for i in range(num_scales * 2)]
        self.num_update = 0
        self.num_scales = num_scales
        self.num_nonzero = [1.0 for i in range(num_scales * 2)]
        self.multiply_factor = 10000

    def update(self, preds, labels):

        for i in range(self.num_scales):
            mask = labels[i * 2]
            label = labels[i * 2 + 1]

            mask_score = torch.index_select(mask, dim=1, index=torch.LongTensor([0, 1])).numpy()
            mask_bbox = torch.index_select(mask, dim=1, index=torch.LongTensor([2, 3, 4, 5])).numpy()

            label_bbox = torch.index_select(label, dim=1, index=torch.LongTensor([2, 3, 4, 5])).numpy()

            pred_score = preds[i * 2].cpu().detach().numpy()
            pred_bbox = preds[i * 2 + 1].cpu().detach().numpy()

            loss_score = numpy.sum(pred_score * mask_score)
            loss_bbox = numpy.sum((label_bbox - pred_bbox) ** 2.0)

            self.num_nonzero[i * 2] += numpy.sum(mask_score[:, 0, :, :] > 0.5)
            self.num_nonzero[i * 2 + 1] += numpy.sum(mask_bbox > 0.5)
            self.sum_metric[i * 2] += loss_score
            self.sum_metric[i * 2 + 1] += loss_bbox

        self.num_update += 1

    def get(self):
        return_string_list = []
        for i in range(self.num_scales):
            return_string_list.append('cls_loss_score_' + str(i))
            return_string_list.append('reg_loss_bbox_' + str(i))

        return return_string_list, [m / self.num_nonzero[i] * self.multiply_factor for i, m in enumerate(self.sum_metric)]

    def reset(self):
        self.sum_metric = [0.0 for i in range(self.num_scales * 2)]
        self.num_update = 0
        self.num_nonzero = [1.0 for i in range(self.num_scales * 2)]
