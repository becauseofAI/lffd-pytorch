# -*- coding: utf-8 -*-
import logging


def start_train(param_dict,
                task_name,
                torch_module,
                gpu_id_list,
                train_dataiter,
                train_metric,
                train_metric_update_frequency,
                num_train_loops,
                val_dataiter,
                val_metric,
                num_val_loops,
                validation_interval,
                optimizer,
                lr_scheduler,
                net,
                net_initializer,
                loss_criterion,
                pretrained_model_param_path,
                display_interval,
                save_prefix,
                model_save_interval,
                start_index
                ):

    logging.info('PyTorch Version: %s', str(torch_module.__version__))
    logging.info('Training settings:-----------------------------------------------------------------')
    for param_name, param_value in param_dict.items():
        logging.info(param_name + ':' + str(param_value))
    logging.info('-----------------------------------------------------------------------------------')

    # init Solver module-------------------------------------------------------------------------------------
    from .solver_GOCD import Solver

    solver = Solver(
        task_name=task_name,
        torch_module=torch_module,
        trainset_dataiter=train_dataiter,
        net=net,
        net_initializer=net_initializer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        gpu_id_list=gpu_id_list,
        num_train_loops=num_train_loops,
        loss_criterion=loss_criterion,
        train_metric=train_metric,
        display_interval=display_interval,
        val_evaluation_interval=validation_interval,
        valset_dataiter=val_dataiter,
        val_metric=val_metric,
        num_val_loops=num_val_loops,
        pretrained_model_param_path=pretrained_model_param_path,
        save_prefix=save_prefix,
        start_index=start_index,
        model_save_interval=model_save_interval,
        train_metric_update_frequency=train_metric_update_frequency)
    solver.fit()
