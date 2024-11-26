import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import copy

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args, model_config):
        super().__init__(args)
        self.model_config = model_config
        self._network = EaseNet(args, True, modelconfig = model_config)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        
        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1 
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        self._network.backbone.add_adapter_to_list()
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    # (proxy_fc = cls * dim)
    def replace_fc(self, train_loader, data_manager):
        model = self._network
        model = model.eval()
        
        with torch.no_grad():           
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0
            
            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                # only use the diagonal feature, index = -1 denotes using init PTM, index = self._cur_task denotes the last adapter's feature
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (idxs, data, label) = batch

                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    if self.args['memory_per_class'] != 0 and self.args['use_old_data'] == True:
                        b_line = self.all_data_num - self.old_data_num
                        indices = torch.where(idxs >= b_line)
                        embedding_list.append(embedding.cpu()[indices[0].numpy()])
                        label_list.append(label.cpu()[indices[0].numpy()])
                    else:
                        embedding_list.append(embedding.cpu())
                        label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(self.train_dataset_for_protonet.labels)

                # proto2 = model.proxy_fc.weight.data
                # model.fc.weight.data[index * 4 : (index + 1) *4, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto2

                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)

                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

            if self.args['memory_per_class'] != 0 and self.args['use_old_data'] == True and self._cur_task > 0:
                embedding_list = []
                label_list = []
                dataset = data_manager.get_olddata_Dummy(self.old_data, self._cur_task)
                # dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train", mode="test", )
                loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=num_workers)
                for i, batch in enumerate(loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                
                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    # print('adapter index:{}, Replacing...{}'.format(self._cur_task, class_index))
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto                
        
        if self.use_diagonal or self.use_exemplars:
            return
        
        if self.recalc_sim:
            self.solve_sim_reset()
        else:
            self.solve_similarity()
    
    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        
        start_cls, end_cls = self.get_cls_range(task_id)
        
        # W(Ti)  i is the i-th task index, T is the cur task index, W is a T*T matrix
        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        # W(TT)
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        # W(ii)
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]
        
        return A.cpu(), B.cpu(), A_hat.cpu()
    
    def solve_similarity(self):       
        for task_id in range(self._cur_task):          
            # print('Solve_similarity adapter:{}'.format(task_id))
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)
            
            # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
            
            # softmax the similarity, it will be failed if not use it
            similarity = F.softmax(similarity, dim=1)
                        
            # weight the combination of B(new_cls2)
            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]
            '''
            A_array = A.numpy()
            B_array = B.numpy()
            A_ext = np.hstack((A_array, np.ones((A_array.shape[0], 1))))
            W_X, _, _, _ = np.linalg.lstsq(A_ext, B_array, rcond=None)
            W = W_X[:-1]
            X = W_X[-1]
            B_hat = torch.matmul(A_hat, torch.tensor(W, dtype=torch.float))+ torch.tensor(X, dtype=torch.float)
            '''
            '''
            A_array = A.numpy()
            A_hat_array = A_hat.numpy()
            A_ext = np.hstack((A_array, np.ones((A_array.shape[0], 1))))
            W_X, _, _, _ = np.linalg.lstsq(A_ext, A_hat_array , rcond=None)
            W = W_X[:-1]
            X = W_X[-1]
            B_hat = torch.matmul(B, torch.tensor(W, dtype=torch.float)) + torch.tensor(X, dtype=torch.float)
            '''
            # B_hat(old_cls2)
            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)
    
    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break
            
            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break
                # print('Solve_similarity adapter:{}, {}'.format(task_id, dim_id))
                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim
                
                # Use the element above the diagonal to calculate
                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()
                
                # calculate similarity matrix between A_hat(old_cls1) and A(new_cls1).
                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)
                
                # softmax the similarity, it will be failed if not use it
                similarity = F.softmax(similarity, dim=1) # dim=1, not dim=0
                            
                # weight the combination of B(new_cls2)
                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]
                
                # B_hat(old_cls2)
                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)
        
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.last_cls = data_manager.get_task_size(-1)
        self._network.nb_tasks = data_manager.nb_tasks
        self._network.update_fc(self._total_classes, inc = data_manager.get_task_size(self._cur_task), use_exemplars = self.use_exemplars)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager

        if self.args['memory_per_class'] != 0 and self.args['use_old_data'] == True:
            print('Using Old Data')
            self.old_data = data_manager.get_olddata(per_class_num=self.args['memory_per_class'])
            self.old_data_num = np.concatenate(self.old_data[0][:self._cur_task + 1]).shape[0]
        else:
            print('Not Using Old Data')
            self.old_data = None
            self.old_data_num = 0

        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train", old_data = self.old_data, cur_task = self._cur_task)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.all_data_num = self.train_loader.dataset.labels.shape[0]

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test", old_data = self.old_data, cur_task = self._cur_task)
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.replace_fc(self.train_loader_for_protonet, data_manager)

        # cnn_accy, nme_accy  = self._network.eval_task()

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']
        
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            ce_losses = 0.0
            reg_losses = 0.0
            reg_losses_std = 0.0
            correct, total = 0, 0
            for i, (idxs, inputs, targets) in enumerate(train_loader):
                embedding_list, label_list = [], []
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                aux_targets = targets.clone()
                if self.args['memory_per_class'] != 0 and self.args['use_old_data'] == True:
                    b_line = self.all_data_num - self.old_data_num
                    indices = torch.where(idxs >= b_line)
                else:
                    aux_targets = torch.where(
                        aux_targets - self._known_classes >= 0,
                        aux_targets - self._known_classes,
                        -1,
                    )
                
                output = self._network(inputs, test=False)
                logits = output["logits"]

                # modified by Qi
                reg_weight =self._network.proxy_fc.weight.data
                reg_embed = torch.zeros_like(reg_weight)

                if self.args['memory_per_class'] != 0 and self.args['use_old_data'] == True:
                    embedding_list.append(output["features"][indices[0].numpy()])
                    label_list.append(aux_targets[indices[0].numpy()])
                else:
                    embedding_list.append(output["features"])
                    label_list.append(aux_targets)

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                class_list = np.unique(label_list.cpu())
                loss_std = torch.tensor(0).cuda()
                for class_index in class_list:
                    # print('adapter index:{}, Replacing...{}'.format(self._cur_task, class_index))
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    reg_embed[class_index]= proto
                    if len(data_index) > 2:
                        loss_std = loss_std + embedding.var(0).sum()
                mask = (reg_embed.sum(dim=1) != 0).float().view(-1, 1)
                reg = ((reg_weight - reg_embed) ** 2 * mask).mean()
                ####

                loss_ce = F.cross_entropy(logits, aux_targets)
                loss_reg = 0.001*reg
                loss_std = 0.0001* loss_std
                loss = loss_ce + loss_reg# + loss_std

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                ce_losses += loss_ce.item()
                reg_losses += loss_reg.item()
                reg_losses_std += loss_std.item()
                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => L {:.3f}, Lce {:.3f}, reg {:.3f}, std {:.3f},T_acc {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    ce_losses / len(train_loader),
                    reg_losses / len(train_loader),
                    reg_losses_std / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]          
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        calc_task_acc = True
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
                
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]