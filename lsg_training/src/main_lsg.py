# The formal version of LSG
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

from models.classifier import Classifier
from models.method import LSG
from tensorboardX import SummaryWriter
from src.utils import load_network, load_data
from src.classnames import *
from src.classnames2 import *
from src.imagenet_templates import IMAGENET_TEMPLATES

def test(loader, model, classifier, device):
    with torch.no_grad():
        model.eval()
        classifier.eval()
        start_test = True
        val_len = len(loader['test'])
        iter_val = iter(loader['test'])
        for _ in range(val_len):
            data = iter_val.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)
            if start_test:
                all_outputs = outputs.data.float()
                all_labels = labels.data.float()
                start_test = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.data.float()), 0)
                all_labels = torch.cat((all_labels, labels.data.float()), 0)
        _, predict = torch.max(all_outputs, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_labels).item() / float(all_labels.size()[0])
    return accuracy

# naive pseudo-labeling by prediction, used for SSL setting.
def pseudo_labeling(loader, model, classifier, device):
    with torch.no_grad():
        start_test = True
        iter_val = iter(loader['unlabeled_test'])
        for i in range(len(loader['unlabeled_test'])):
            data = iter_val.next()
            inputs = data[0][0]
            labels = data[1]
            indexs = data[2]
            inputs = inputs.to(device)
            labels = labels.to(device)
            feat = model.inference(inputs)
            outputs = classifier(feat)

            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                all_index = indexs.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
                all_index = torch.cat((all_index, indexs.data.float()), 0)

        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        # print(predict, all_label)
    return accuracy, predict

def train(args, text_features, model, classifier, dataset_loaders, optimizer, scheduler, device=None, writer=None, model_path = None):

    len_labeled = len(dataset_loaders["train"])
    iter_labeled = iter(dataset_loaders["train"])

    len_unlabeled = len(dataset_loaders["unlabeled_train"])
    iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

    n_classes, n_prompts, _ = text_features.size()
    text_labels = []
    for i in range(n_classes):
        text_labels += [i for j in range(n_prompts)]
    text_labels = torch.tensor(text_labels).cuda()

    criterions = {"CrossEntropy": nn.CrossEntropyLoss(), "KLDiv": nn.KLDivLoss(reduction='batchmean')}
    
    best_acc = 0.0
    best_model = None

    pseudo_labels = None
    cls = 0.
    aln = [0.,0.]
    reg = [0.,0.]
    for iter_num in range(1, args.max_iter + 1):
        if (iter_num-1) % len_unlabeled == 0:
            model.eval()
            classifier.eval()
            pacc, pseudo_labels = pseudo_labeling(dataset_loaders, model, classifier, device=device)
            if args.cu > 0:
                print('Iter: {}, pseudo-labeling acc: {:.2f}'.format(iter_num, pacc))
    
        model.train(True)
        classifier.train(True)
        optimizer.zero_grad()
        if iter_num % len_labeled == 0:
            iter_labeled = iter(dataset_loaders["train"])
        if iter_num % len_unlabeled == 0:
            iter_unlabeled = iter(dataset_loaders["unlabeled_train"])

        data_labeled = iter_labeled.next()
        data_unlabeled = iter_unlabeled.next()

        img_labeled_q = data_labeled[0][0].to(device)
        img_labeled_k = data_labeled[0][1].to(device)
        label = data_labeled[1].to(device)

        img_unlabeled_q = data_unlabeled[0][0].to(device)
        img_unlabeled_k = data_unlabeled[0][1].to(device)
        index = data_unlabeled[2]
        plabel = pseudo_labels[index].to(device)

        ## For Labeled Data
        label_expand = torch.cat((label, label),dim=0)
        img_labeled = torch.cat([img_labeled_q,img_labeled_k],dim=0)
        emb_labeled, emb_graph_labeled, nodefeat_labeled, feat_labeled = model(img_labeled, label_expand)
        
        out = classifier(feat_labeled)
        classifier_loss = criterions['CrossEntropy'](out, label_expand)
        cls += classifier_loss.item()
 
        align_loss_labeled = criterions['CrossEntropy'](emb_labeled, label_expand) # L_align
        emb_graph_labeled = F.normalize(emb_graph_labeled, dim=-1)
        nodefeat_labeled = F.normalize(nodefeat_labeled, dim=-1)
        reg_loss_labeled= (nodefeat_labeled-emb_graph_labeled.detach()).norm(p=2,dim=-1,keepdim=True).mean() # L_r

        _, graph_predict = emb_labeled.detach().max(dim=1)
        graph_accuracy = torch.sum(torch.squeeze(graph_predict).float() == label_expand).item() / float(label_expand.size(0))

        aln[0] += align_loss_labeled.item()*args.cg
        reg[0] += reg_loss_labeled.item()*args.cc
        semantic_loss_labeled = align_loss_labeled*args.cg + reg_loss_labeled*args.cc

        ## For Unlabeled Data
        plabel_expand = torch.cat((plabel, plabel),dim=0)
        img_unlabeled = torch.cat([img_unlabeled_q,img_unlabeled_k],dim=0)
        emb_unlabeled, emb_graph_unlabeled, nodefeat_unlabeled, feat_unlabeled = model(img_unlabeled, plabel_expand)

        align_loss_unlabeled = criterions['CrossEntropy'](emb_unlabeled, plabel_expand) 
        emb_graph_unlabeled = F.normalize(emb_graph_unlabeled,dim=-1)
        nodefeat_unlabeled = F.normalize(nodefeat_unlabeled,dim=-1)
        reg_loss_unlabeled = (nodefeat_unlabeled-emb_graph_unlabeled.detach()).norm(p=2,dim=-1,keepdim=True).mean()

        aln[1] += align_loss_unlabeled.item()*args.cg*args.cu
        reg[1] += reg_loss_unlabeled.item()*args.cc*args.cu
        semantic_loss_unlabeled = align_loss_unlabeled*args.cg + reg_loss_unlabeled*args.cc

        total_loss = classifier_loss + semantic_loss_labeled + semantic_loss_unlabeled*args.cu
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        ## Calculate the training accuracy of current iteration
        if iter_num % 100 == 0:
            _, predict = torch.max(out, 1)
            hit_num = (predict == label_expand).sum().item()
            sample_num = predict.size(0)
            hit_num2 = (graph_predict == label_expand).sum().item()
            sample_num2 = graph_predict.size(0)
            print("Iter: {}; main model acc: {:.4f}; graph acc {:.4f}".format(iter_num, hit_num / float(sample_num), hit_num2 / float(sample_num2)))
            print("cls_loss:{:.4f}, align_loss:{:.4f}/{:.4f}, reg_loss:{:.4f}/{:.4f}\n".format(cls/100,aln[0]/100,aln[1]/100,reg[0]/100,reg[1]/100))
            cls = 0.
            aln = [0.,0.]
            reg = [0.,0.]

        ## Show Loss in TensorBoard
        writer.add_scalar('loss/classifier_loss', classifier_loss, iter_num)
        writer.add_scalar('loss/align_loss_labeled', align_loss_labeled, iter_num)
        writer.add_scalar('loss/reg_loss_labeled', reg_loss_labeled, iter_num)
        writer.add_scalar('loss/total_loss', total_loss, iter_num)

        if iter_num % args.test_interval == 1:# or iter_num == 500:
            if iter_num == 1:
               continue
            model.eval()
            classifier.eval()
            test_acc = test(dataset_loaders, model, classifier, device=device)
            print("iter_num: {}; test_acc: {}".format(iter_num, test_acc))
            writer.add_scalar('acc/test_acc', test_acc, iter_num)
            if test_acc > best_acc:
                best_acc = test_acc
                best_model = {'model': model.state_dict(),
                              'classifier': classifier.state_dict(),
                              'step': iter_num
                              }
    print("best acc: %.4f" % (best_acc))
    torch.save(best_model, model_path)
    print("The best model has been saved in ", model_path)

def text_embedding(filename):
    all_text_features = torch.load(filename)
    all_text_features = all_text_features[:,:20,:]
    print("Label embeddings loaded with size:", all_text_features.size())
    return all_text_features

def read_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='root path of dataset')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--label_ratio', type=int, default=15)
    parser.add_argument('--logdir', type=str, default='../vis/')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--cu', type=float, default=0.0) # no unlabeled data used
    parser.add_argument('--cc', type=float, default=1.0)
    parser.add_argument('--cg', type=float, default=1.0)
    parser.add_argument('--w_ti', type=float, default=1.0)
    parser.add_argument('--w_ii', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr_ratio', type=float, default=10)
    # parser.add_argument('--lr_gamma', type=float, default=0.0002)
    # parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--weight_decay', type=float, default=0.0001)

    parser.add_argument('--projector_dim', type=int, default=1024)
    parser.add_argument('--class_num', type=int, default=200)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--max_iter', type=int, default=27005)
    parser.add_argument('--test_interval', type=float, default=3000)
    parser.add_argument("--pretrained", action="store_true", help="use the pre-trained model")
    parser.add_argument("--gcn_path", type=str, default='gcn.pt')

    parser.add_argument("--text_model_name", type=str, default='bert')
    parser.add_argument('--edge_ratio', type=float, default=0.003, help='ratio of edges in text graph')
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
    args.data_return_index = True

    dataset_loaders = load_data(args)
    print("dataset: {}, class_num: {}".format(args.dataset,args.class_num))

    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

    model_name = "%s_%s_%s" % (args.backbone, os.path.basename(args.root), str(args.label_ratio))
    logdir = os.path.join(args.logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(logdir)

    names = {'clip':'_clip.pt', 'clipL':'_clipL.pt','bert':'_BertL.pt', 'mT5L':'_mT5L.pt'}
    filename = '../label_embedding/' + args.dataset + names[args.text_model_name]
    print('Using pretrained label languistic embeddings: ',filename)
    text_features = text_embedding(filename)

    model_path = os.path.join(logdir, "%s_best.pkl" % (model_name))

    # Initialize model
    network, feature_dim = load_network(args.backbone)

    model = LSG(network=network, backbone=args.backbone, text_features=text_features, class_num=args.class_num, pretrained=args.pretrained, edge_ratio=args.edge_ratio, w_ti=args.w_ti, w_ii=args.w_ii).to(device)
    classifier = Classifier(feature_dim, args.class_num).to(device)

    dataset_str = args.root.split('/')[-1] 
    gcn_path = args.gcn_path 
    model.load_pretrained_gcn(gcn_path)
    print("load pretrained gcn weight from "+gcn_path+'\n')

    print("backbone params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1e6 / 2))
    print("classifier params: {:.2f}M".format(sum(p.numel() for p in classifier.parameters()) / 1e6))
    
    
    ## Define Optimizer
    base_params = []
    for pname, p in model.named_parameters():
        if pname.startswith('encoder.gnn'):
            # print(pname)
            p.requires_grad=False
        else:
            base_params += [p]
    optimizer = optim.SGD([
        {'params': base_params},
        {'params': classifier.parameters(), 'lr': args.lr * args.lr_ratio},
    ], lr= args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    milestones = [6000, 12000, 18000, 24000]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    # scheduler = LambdaLR(optimizer, lambda x: (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # Train model
    print('\n******************* Start Training *******************')
    print(args)
    train(args, text_features, model, classifier, dataset_loaders, optimizer, scheduler, device=device, writer=writer, model_path=model_path)

if __name__ == '__main__':
    main()
