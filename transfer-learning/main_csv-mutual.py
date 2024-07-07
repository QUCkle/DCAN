import configargparse
import data_loader
import matplotlib.pyplot as plt
import os
import time
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
import argparse
from torchvision import transforms
from scripts import CSVDataset, ToTensor
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)
    
    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False, help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=50, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=0.5)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


   
def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    # 总共三个数据集，分别是源域数据集、目标域训练数据集和目标域测试数据集
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    folder_test = os.path.join(args.data_dir, 'test')
    source_loader, n_class,source_data = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ ,target_train_data= data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ ,target_test_data= data_loader.load_data(
        folder_test, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    
    
    # print(source_loader.dataset[0].shape)
    # print(target_train_loader.dataset[0].shape)
    # print(target_test_loader.dataset[0].shape)
    return source_loader, target_train_loader, target_test_loader, n_class,source_data,target_train_data,target_test_data

# 获取模型参数，模型参数包括类别数、迁移损失、基础网络、最大迭代次数、是否使用瓶颈层
def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model

# 获取优化器，优化器参数包括模型和参数
def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

# 获取学习率调度器，学习率调度器参数包括优化器和参数
def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler


# 测试函数，输入是目标域的数据，输出是准确率和损失
def test(model,target_test_loader, args,epoch,domain):
    featrues_all = pd.DataFrame()
    label_all = pd.DataFrame()
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    all_labels = []
    all_preds = []

    with torch.no_grad():
        iter_target_test = iter(target_test_loader)
        while True:
            try:
                data_1 = next(iter_target_test)
                data, label = data_1['data'], data_1['label']
                data, label = data.to(args.device), label.to(args.device)
                data = data.unsqueeze(1)
                data = data.unsqueeze(2)
                s_output,featrues = model.predict(data)
                featrues = featrues.cpu().numpy()
                featrues = pd.DataFrame(featrues)
                featrues_all = pd.concat([featrues_all,featrues],axis=0)
                label_all = pd.concat([label_all,pd.DataFrame(label.cpu().numpy())],axis=0)
                loss = criterion(s_output, label)
                test_loss.update(loss.item())
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == label)
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
            except StopIteration:
                break
            
        featrues_all = pd.concat([featrues_all,label_all],axis=1)
        featrues_all.to_csv(f"/home/Q/Diploma_thesis/transferlearning-master-CAE/code/DeepDA/train_log/features/{domain}/{epoch}.csv",index=False)

    acc = 100. * correct / len_target_dataset

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Create directory for saving confusion matrix and data
    output_dir = f"/home/Q/Diploma_thesis/transferlearning-master-CAE/code/DeepDA/train_log/{domain}/confusion/{epoch}"
    os.makedirs(output_dir, exist_ok=True)

    # Save confusion matrix data to csv file
    np.savetxt(f"{output_dir}/confusion_matrix.csv", cm, delimiter=",")

    # Plot confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.savefig(f"{output_dir}/confusion_matrix.png")

    return acc, test_loss.avg




# 主要的训练函数，包括源域数据、目标域训练数据、目标域测试数据、模型、优化器、学习率调度器和参数
def train(source_loader, target_train_loader, source_test_loader,target_test_loader, model, optimizer, lr_scheduler, args):
    
    my_log = []
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch 
    
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    
    for e in range(1, args.n_epoch+1):
        
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)
        
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(n_batch):
            

            data_1 = next(iter_source)
            data_source, label_source = data_1['data'], data_1['label']
            data_2 = next(iter_target)
            data_target, _ = data_2['data'], data_2['label']
            
            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)
            data_target = data_target.to(args.device)

         
            # 源域数据
            # 升维
            data_source = data_source.unsqueeze(1)      
            data_source = data_source.unsqueeze(1)
            # print(data_source.size)
            
            
            # 目标域数据 
            # 升维
            data_target = data_target.unsqueeze(1)
            data_target = data_target.unsqueeze(1)
            # 这里的model()函数就是model中的forword函数
            
            
            # clf_loss是在源域上的标签预测损失，transfer_loss是迁移损失也就是域分类的损失
            clf_loss, transfer_loss = model(data_source, data_target, label_source)
            loss = clf_loss + args.transfer_loss_weight * transfer_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if lr_scheduler:
                lr_scheduler.step()

            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            # print(n_batch,batch_idx,train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)


        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.6f}, total_Loss: {:.4f}'.format(
            e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)

        stop += 1
        
        # # 在train函数中调用test函数的部分
        # test_features_tensor = torch.tensor(target_test_features).float()
        # test_labels_tensor = torch.tensor(target_test_labels).long()
        # # 升维
        # test_features_tensor = test_features_tensor.unsqueeze(1)
        # test_features_tensor = test_features_tensor.unsqueeze(2)
        
        # test_loss是在目标域上的标签预测损失，test_acc是准确率
        test_acc_tgt, test_loss_tgt = test(model,target_test_loader, args,e,domain="tgt")
        test_acc_src, test_loss_src = test(model,source_test_loader, args,e,domain="src")
        
        # test_acc, test_loss = test(model, torch.tensor(target_test_features).float(), torch.tensor(target_test_labels).long(), args)
        
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg,test_loss_tgt, test_acc_tgt.item(),test_loss_src, test_acc_src.item()])
        info += ', test_loss_tgt {:4f}, test_acc_tgt: {:.4f}'.format(test_loss_tgt, test_acc_tgt)
        info += ', test_loss_src {:4f}, test_acc_src: {:.4f}'.format(test_loss_src, test_acc_src)

        np_log = np.array(log, dtype=float)
        np.savetxt('/home/Q/Diploma_thesis/transferlearning-master-CAE/code/DeepDA/train_log.csv', np_log, delimiter=',', fmt='%.6f')
        

        
        if best_acc < test_acc_tgt:
            best_acc = test_acc_tgt
            stop = 0

        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        
        print(info)
    log_path = '/home/Q/Diploma_thesis/transferlearning-master-CAE/code/DeepDA/train_log'
    
    np.savetxt(os.path.join(log_path, 'train_log.csv'), np_log, delimiter=',', fmt='%.6f')

    # Plot the training curves
    plt.figure(figsize=(10, 6))
    plt.plot(np_log[:, 0], label='cls_loss')
    plt.plot(np_log[:, 1], label='transfer_loss')
    plt.plot(np_log[:, 2], label='total_loss')
    plt.plot(np_log[:, 3], label='test_loss')
    plt.plot(np_log[:, 4], label='test_acc')    
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_path, 'training_curves.png'))
    print('Transfer result: {:.4f}'.format(best_acc))

def main():
    '''
    # 创建一个 ArgumentParser 对象
    其中模型config的参数如下：
    1、基于对抗的迁移学习方法：
    DANN是传统的迁移学习方法，loss=源域标签损失+目标域标签损失+领域分类损失(有梯度反转)
    DAAN (Dynamic Adversarial Adaptation Network)一种可以动态调整边界和条件分布关系的深度对抗网络模型，
    它的基础网络与DANN网络基本一致，核心在于引入了条件域判别块和集成化的动态调节因子ω(lamda)，用于调节全局对抗损失和局部对抗损失的权重。
    2、基于最大均值差异的迁移学习方法：
    DAN (Deep Adaptation Network)是一种基于最大均值差异的迁移学习方法，它的基础网络与DANN网络基本一致，核心在于引入了最大均值差异损失。
    他的loss=源域标签损失+目标域标签损失+最大均值差异损失。最大均值差异损失是源域和目标域的特征分布的差异。
    '''

    parser = argparse.ArgumentParser()
    # 添加参数，参照文件夹中的.yaml文件，直接修改defalut参数，我这边没有自动读取
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--batch_size', type=int, default= 64)
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--early_stop', type=int, default=0)
    parser.add_argument('--epoch_based_training', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_scheduler', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--n_iter_per_epoch', type=int, default=500)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--src_domain', type=str, default='source')
    parser.add_argument('--tgt_domain', type=str, default='target')
    parser.add_argument('--transfer_loss', type=str, default='daan')
    parser.add_argument('--transfer_loss_weight', type=float, default=5)
    parser.add_argument('--use_bottleneck', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # 解析参数
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    set_random_seed(args.seed)
    # source_loader, target_train_loader, target_test_loader, n_class,source_data,target_train_data,target_test_data = load_data(args)
        # 定义转换操作
    transform = transforms.Compose([
        ToTensor()  # 转换为张量
    ])
    
    # RAW
    
    # folder_src = "/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k/source/EN_train_balanced.csv"
    # folder_tgt = "/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k/target/ZH_train.csv"
    # folder_test_tgt = "/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k/ZH_test.csv"  
    # folder_test_src = "/home/Q/dataset/transfer-learning/trans-language/raw_resample/1k/EN_test_balanced.csv"  
    
    
    # # CAE
    # folder_src = "/home/Q/dataset/transfer-learning/trans-language/CAE/1k/source/EN_train_balanced.csv"
    # folder_tgt = "/home/Q/dataset/transfer-learning/trans-language/CAE/1k/target/ZH_train.csv"
    # folder_test_tgt = "/home/Q/dataset/transfer-learning/trans-language/CAE/1k/ZH_test.csv"
    # folder_test_src = "/home/Q/dataset/transfer-learning/trans-language/CAE/1k/EN_test_balanced.csv"        
    
    # manual
    folder_src = "/home/Q/dataset/transfer-learning/trans-language/手动/source/EN_train.csv"
    folder_tgt = "/home/Q/dataset/transfer-learning/trans-language/手动/target/ZH_train.csv"
    folder_test_tgt = "/home/Q/dataset/transfer-learning/trans-language/手动/test/ZH_test_updated_2.csv"
    folder_test_src = "/home/Q/dataset/transfer-learning/trans-language/手动/EN_test.csv"      
    
    dataset_src = CSVDataset(csv_file=folder_src, transform=transform)
    dataset_tgt = CSVDataset(csv_file=folder_tgt, transform=transform)
    dataset_test_src = CSVDataset(csv_file=folder_test_src, transform=transform)
    dataset_test_tgt = CSVDataset(csv_file=folder_test_tgt, transform=transform)
    
    source_loader = torch.utils.data.DataLoader(dataset=dataset_src, batch_size=args.batch_size, shuffle=True)
    target_train_loader = torch.utils.data.DataLoader(dataset=dataset_tgt, batch_size=args.batch_size, shuffle=True)
    target_test_loader = torch.utils.data.DataLoader(dataset=dataset_test_tgt, batch_size=args.batch_size, shuffle=True)
    source_test_loader = torch.utils.data.DataLoader(dataset=dataset_test_src, batch_size=args.batch_size, shuffle=True)
    n_class = 2     
    
    setattr(args, "n_class", n_class)
    # args.epoch_based_training决定了是基于epoch还是基于iteration的训练
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        #　setattr是设置属性的函数，这里设置了最大迭代次数
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    # print(model)

    optimizer = get_optimizer(model, args)
    
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
        
    # 训练
    train(source_loader, target_train_loader, source_test_loader,target_test_loader, model, optimizer, scheduler, args)
    

if __name__ == "__main__":
    main()
