import sys
import os
proj_dir = os.path.abspath(os.getcwd())
print("proj_dir: ", proj_dir)
sys.path.append(proj_dir)

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.gcn import GCN
from tensorboardX import SummaryWriter

from src.classnames import *
from src.classnames2 import *

def train(args, text_features, text_features_test, model, optimizer, scheduler, device=None, writer=None, model_path = None):

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}

    n_classes, n_prompts, _ = text_features.size()
    text_labels = []
    for i in range(n_classes):
        text_labels += [i for j in range(n_prompts)]
    text_labels = torch.tensor(text_labels).cuda()

    n_classes, n_prompts_test, _ = text_features_test.size()
    text_labels_test = []
    for i in range(n_classes):
        text_labels_test += [i for j in range(n_prompts_test)]
    text_labels_test = torch.tensor(text_labels_test).cuda()

    best_acc = 0.0
    best_model = None

    text_input = text_features.reshape(-1,text_features.size(-1)).to(device)
    cls = 0.
    for iter_num in range(1, args.max_iter + 1):
        model.train(True)
        optimizer.zero_grad()
        
        text_output = model(text_input)
        loss = criterions['CrossEntropy'](text_output, text_labels)
        _, pred = text_output.detach().max(dim=1)
        
        accuracy = torch.sum(torch.squeeze(pred).float() == text_labels).item() / float(text_labels.size(0))
        # print("iter:{}, acc{:.2f}, loss{:.4f}".format(iter_num, accuracy*100, loss.item()))

        loss.backward()
        optimizer.step()
        scheduler.step()
        cls += loss.item()

        ## Calculate the training accuracy of current iteration
        if iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                text_input_test = text_features_test.reshape(-1,text_features_test.size(-1)).cuda()
                text_output_test = model.inference(text_input_test)
                _, predict = torch.max(text_output_test, 1)
                hit_num = (predict == text_labels_test).sum().item()
                sample_num = predict.size(0)
                acc = hit_num / sample_num
                if acc > best_acc:
                    best_acc = acc
                print("Iter: {}; current acc: {}".format(iter_num, hit_num / float(sample_num)))
            model.train()
            print("cls_loss:{:.4f}\n".format(cls/100))
            cls = 0.

        writer.add_scalar('loss/total_loss', loss, iter_num)

    best_model = model.state_dict()
    print("best accuracy: {:.2f}".format(best_acc*100))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def text_embedding(filename):
    all_text_features = torch.load(filename) # n_classes * n_prompts * dimension
    tr_text_features = all_text_features[:,:20,:]
    print("train feature size:", tr_text_features.size())
    te_text_features = all_text_features[:,20:40,:]
    print("test feature size:", te_text_features.size())
    return tr_text_features, te_text_features

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default='0.001')
    parser.add_argument('--seed', type=int, default='666666')
    parser.add_argument('--workers', type=int, default='4')
    parser.add_argument('--lr_ratio', type=float, default='10')
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=100)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=5005)
    parser.add_argument('--test_interval', type=float, default=100)

    parser.add_argument("--text_model_name", type=str, default='bert')
    parser.add_argument('--edge_ratio', type=float, default=0.002, help='ratio of edges in text graph')
    configs = parser.parse_args()
    return configs

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    args = read_config()
    set_seed(args.seed)

    if 'CUB200' in args.root:
        args.dataset = 'bird'
        args.class_num = 200
        args.classname = CLASSES_Birds
    elif 'StanfordCars' in args.root:
        args.dataset = 'cars'
        args.class_num = 196
        args.classname = CLASSES_Car
    elif 'Aircraft' in args.root:
        args.dataset = 'aircraft'
        args.class_num = 100
        args.classname = CLASSES_AirCraft
    else:
        raise NotImplementedError

    print("class_num: ", args.class_num)

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    logdir = os.path.join(args.logdir, args.dataset)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    names = {'clip':'_clip.pt', 'clipB':'_clipB.pt','bert':'_BertL.pt', 'mT5L':'_mT5L.pt'}
    filename = '../label_embedding/' + args.dataset + names[args.text_model_name]

    print('using text model: ',filename)
    tr_text_features, te_text_features = text_embedding(filename)
    tr_text_features = tr_text_features.float()
    te_text_features = te_text_features.float()

    model_path = os.path.join(logdir, "gcn-%s.pkl" % (args.dataset))

    model = GCN(tr_text_features,ratio=args.edge_ratio)
    model.to(device)

    ## Define Optimizer
    optimizer = optim.SGD([
        {'params': model.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    # Train model
    print(args)
    train(args, tr_text_features, te_text_features, model, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

if __name__ == '__main__':
    main()
