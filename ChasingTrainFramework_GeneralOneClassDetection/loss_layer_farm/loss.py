# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class cross_entropy_with_hnm_for_one_class_detection2(nn.Module):
    def __init__(self, hnm_ratio, num_output_scales):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)
        self.num_output_scales = num_output_scales

    def forward(self, outputs, targets):
        loss_branch_list = []
        for i in range(self.num_output_scales):
            pred_score = outputs[i * 2]
            pred_bbox = outputs[i * 2 + 1]
            gt_mask = targets[i * 2].cuda()
            gt_label = targets[i * 2 + 1].cuda()

            pred_score_softmax = torch.softmax(pred_score, dim=1)
            # loss_mask = torch.ones(pred_score_softmax.shape[0],
            #                        1,
            #                        pred_score_softmax.shape[2],
            #                        pred_score_softmax.shape[3])
            loss_mask = torch.ones(pred_score_softmax.shape)

            if self.hnm_ratio > 0:
                # print('gt_label.shape:', gt_label.shape)
                # print('gt_label.size():', gt_label.size())
                pos_flag = (gt_label[:, 0, :, :] > 0.5)
                pos_num = torch.sum(pos_flag) # get num. of positive examples

                if pos_num > 0:
                    neg_flag = (gt_label[:, 1, :, :] > 0.5)
                    neg_num = torch.sum(neg_flag)
                    neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                    # non-negative value
                    neg_prob = torch.where(neg_flag, pred_score_softmax[:, 1, :, :], \
                                           torch.zeros_like(pred_score_softmax[:, 1, :, :]))
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)

                    prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                    neg_grad_flag = (neg_prob <= prob_threshold)
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
                else:
                    neg_choice_ratio = 0.1
                    neg_num_selected = int(pred_score_softmax[:, 1, :, :].numel() * neg_choice_ratio)
                    neg_prob = pred_score_softmax[:, 1, :, :]
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                    neg_grad_flag = (neg_prob <= prob_threshold)
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)

            # cross entropy with mask
            pred_score_softmax_masked = pred_score_softmax[loss_mask]
            pred_score_log = torch.log(pred_score_softmax_masked)
            score_cross_entropy = -gt_label[:, :2, :, :][loss_mask] * pred_score_log
            loss_score = torch.sum(score_cross_entropy) / score_cross_entropy.numel()

            mask_bbox = gt_mask[:, 2:6, :, :]
            if torch.sum(mask_bbox) == 0:
                loss_bbox = torch.zeros_like(loss_score)
            else:
                predict_bbox = pred_bbox * mask_bbox
                label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
                loss_bbox = F.mse_loss(predict_bbox, label_bbox, reduction='mean')
                # loss_bbox = F.smooth_l1_loss(predict_bbox, label_bbox, reduction='mean')
                # loss_bbox = torch.nn.MSELoss(predict_bbox, label_bbox, size_average=True, reduce=True)
                # loss_bbox = torch.nn.SmoothL1Loss(predict_bbox, label_bbox, size_average=True, reduce=True)

            loss_branch = loss_score + loss_bbox
            loss_branch_list.append(loss_branch)
        return loss_branch_list


class cross_entropy_with_hnm_for_one_class_detection(nn.Module):
    def __init__(self, hnm_ratio, num_output_scales):
        super(cross_entropy_with_hnm_for_one_class_detection, self).__init__()
        self.hnm_ratio = int(hnm_ratio)
        self.num_output_scales = num_output_scales

    def forward(self, outputs, targets):
        loss_cls = 0
        loss_reg = 0
        loss_branch = []
        for i in range(self.num_output_scales):
            pred_score = outputs[i * 2]
            pred_bbox = outputs[i * 2 + 1]
            gt_mask = targets[i * 2].cuda()
            gt_label = targets[i * 2 + 1].cuda()

            pred_score_softmax = torch.softmax(pred_score, dim=1)
            # loss_mask = torch.ones(pred_score_softmax.shape[0],
            #                        1,
            #                        pred_score_softmax.shape[2],
            #                        pred_score_softmax.shape[3])
            loss_mask = torch.ones(pred_score_softmax.shape)

            if self.hnm_ratio > 0:
                # print('gt_label.shape:', gt_label.shape)
                # print('gt_label.size():', gt_label.size())
                pos_flag = (gt_label[:, 0, :, :] > 0.5)
                pos_num = torch.sum(pos_flag) # get num. of positive examples

                if pos_num > 0:
                    neg_flag = (gt_label[:, 1, :, :] > 0.5)
                    neg_num = torch.sum(neg_flag)
                    neg_num_selected = min(int(self.hnm_ratio * pos_num), int(neg_num))
                    # non-negative value
                    neg_prob = torch.where(neg_flag, pred_score_softmax[:, 1, :, :], \
                                           torch.zeros_like(pred_score_softmax[:, 1, :, :]))
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)

                    prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                    neg_grad_flag = (neg_prob <= prob_threshold)
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)
                else:
                    neg_choice_ratio = 0.1
                    neg_num_selected = int(pred_score_softmax[:, 1, :, :].numel() * neg_choice_ratio)
                    neg_prob = pred_score_softmax[:, 1, :, :]
                    neg_prob_sort, _ = torch.sort(neg_prob.reshape(1, -1), descending=False)
                    prob_threshold = neg_prob_sort[0][neg_num_selected-1]
                    neg_grad_flag = (neg_prob <= prob_threshold)
                    loss_mask = torch.cat([pos_flag.unsqueeze(1), neg_grad_flag.unsqueeze(1)], dim=1)

            # cross entropy with mask
            pred_score_softmax_masked = pred_score_softmax[loss_mask]
            pred_score_log = torch.log(pred_score_softmax_masked)
            score_cross_entropy = -gt_label[:, :2, :, :][loss_mask] * pred_score_log
            loss_score = torch.sum(score_cross_entropy) / score_cross_entropy.numel()

            mask_bbox = gt_mask[:, 2:6, :, :]
            if torch.sum(mask_bbox) == 0:
                loss_bbox = torch.zeros_like(loss_score)
            else:
                predict_bbox = pred_bbox * mask_bbox
                label_bbox = gt_label[:, 2:6, :, :] * mask_bbox
                loss_bbox = F.mse_loss(predict_bbox, label_bbox, reduction='sum') / torch.sum(mask_bbox)
                # loss_bbox = F.smooth_l1_loss(predict_bbox, label_bbox, reduction='sum') / torch.sum(mask_bbox)
                # loss_bbox = torch.nn.MSELoss(predict_bbox, label_bbox, size_average=False, reduce=True)
                # loss_bbox = torch.nn.SmoothL1Loss(predict_bbox, label_bbox, size_average=False, reduce=True)

            loss_cls += loss_score
            loss_reg += loss_bbox
            loss_branch.append(loss_score)
            loss_branch.append(loss_bbox)
        loss = loss_cls + loss_reg
        return loss, loss_branch