import torch
import torch.nn as nn
from transfer_losses import TransferLoss
import backbones


class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, max_iter=1000, **kwargs):
        super(TransferNet, self).__init__()
        self.num_class = num_class
        # base_network在迁移学习中是一个预训练的网络，这里使用的是resnet50,担任特征提取器的作用
        self.base_network = backbones.get_backbone(base_net)
        # use_bottleneck是一个布尔值，如果为True，则使用bottleneck层，否则不使用,bottleneck_width是bottleneck层的宽度,bottleneck层的作用是将特征的维度降低
        self.use_bottleneck = use_bottleneck
        # transfer_loss计算方法定义
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            # 如果使用了bottleneck层
            bottleneck_list = [
                # 添加一个全连接层，输入维度为base_network的输出维度，输出维度为bottleneck_width，起到了降维的作用
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            # 初始化bottleneck_layer模型
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            # 压缩后的特征维度就是bottleneck_width
            feature_dim = bottleneck_width
        else:
            # 否则特征维度不变
            feature_dim = self.base_network.output_num()
        
        # 特征分类器,这是一个全连接层，输入维度为feature_dim，输出维度为num_class,这是标签分类器
        self.classifier_layer = nn.Linear(feature_dim, num_class)
        
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        
        # 迁移损失,也是就是在源域和目标域之间的损失，将transfer_lossed.py中的loss函数给到adapt_loss
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        # 定义损失函数为交叉熵函数
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, source, target, source_label):
        # 构建反向传播模型，这里的base_network()就是backbone.py中的forward函数
        # 利用base_network()函数给source和target提取特征
        source = self.base_network(source)
        target = self.base_network(target)
        
        # 如果使用bottleneck层，就对source和target进行特征压缩
        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
            
        # 标签分类
        # source_clf是源域的标签，这里是用来计算分类器的损失，source_clf共有0和1两种情况
        source_clf = self.classifier_layer(source)
        # 计算标签预测损失
        clf_loss = self.criterion(source_clf, source_label)
        
        # 领域分类
        kwargs = {}
        # 下面的if-elif语句是用来判断迁移损失的类型
        # 只有他们三个需要额外的处理
        if self.transfer_loss == "lmmd":
            # lmmd情况下，需要计算源域和目标域的分类器输出
            kwargs['source_label'] = source_label
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            
        elif self.transfer_loss == "daan":
            # daan情况下，需要计算源域和目标域的分类器输出
            source_clf = self.classifier_layer(source)
            # source_logits是源域的分类器输出
            kwargs['source_logits'] = torch.nn.functional.softmax(source_clf, dim=1)
            target_clf = self.classifier_layer(target)
            kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
            
        elif self.transfer_loss == 'bnm':
            # bnm情况下，需要计算源域和目标域的分类器输出
            tar_clf = self.classifier_layer(target)
            target = nn.Softmax(dim=1)(tar_clf)   
            
        transfer_loss = self.adapt_loss(source, target, **kwargs)
        
        return clf_loss, transfer_loss
    
    def get_parameters(self, initial_lr=1.0):
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        return params

    def predict(self, x):
        # 先提取特征，然后再通过分类器进行分类
        features = self.base_network(x)
        # 对特征进行压缩
        x = self.bottleneck_layer(features)
        # 标签分类
        clf = self.classifier_layer(x)
        return clf,x

    # 这个函数的作用是在每个epoch结束后调用，用于更新daan中的动态因子
    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass