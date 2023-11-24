from models.resnet_gcn import resnet18, resnet34, resnet50, resnet152, resnet101

from data.tranforms import TransformTrain, TransformTrainStrong
from data.tranforms import TransformTest
import data
from torch.utils.data import DataLoader, RandomSampler
import os

imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)

def load_data(args):
    batch_size_dict = {"train": args.batch_size, "unlabeled_train": args.batch_size, "test": 100}

    transform_train = TransformTrain() if args.cu == 0 else TransformTrainStrong()
    if 'CUB200' in args.root:
        transform_train = TransformTrainStrong()
    transform_test = TransformTest(mean=imagenet_mean, std=imagenet_std)
    if args.data_return_index:
        dataset = data.__dict__[os.path.basename(args.root)+'_IDX']
    else:
        dataset = data.__dict__[os.path.basename(args.root)]

    datasets = {"train": dataset(root=args.root, split='train', label_ratio=args.label_ratio, download=True, transform=transform_train),
                "unlabeled_train": dataset(root=args.root, split='unlabeled_train', label_ratio=args.label_ratio, download=True, transform=transform_train)}
    test_dataset = {'test': dataset(root=args.root, split='test', label_ratio=100, download=True, transform=transform_test)}
    datasets.update(test_dataset)

    dataset_loaders = {x: DataLoader(datasets[x], batch_size=batch_size_dict[x], shuffle=True, num_workers=4, drop_last=True)
                        for x in ['train', 'unlabeled_train']}
    dataset_loaders.update({'unlabeled_test': DataLoader(datasets["unlabeled_train"], batch_size=batch_size_dict['test'], shuffle=False, num_workers=4)})
    dataset_loaders.update({'test': DataLoader(datasets["test"], batch_size=batch_size_dict['test'], shuffle=False, num_workers=4)})


    return dataset_loaders

def load_network(backbone):
    if 'resnet' in backbone:
        if backbone == 'resnet18':
            network = resnet18
            feature_dim = 512
        elif backbone == 'resnet34':
            network = resnet34
            feature_dim = 512
        elif backbone == 'resnet50':
            network = resnet50
            feature_dim = 2048
        elif backbone == 'resnet101':
            network = resnet101
            feature_dim = 2048
        elif backbone == 'resnet152':
            network = resnet152
            feature_dim = 2048
    else:
        raise NotImplementedError

    return network, feature_dim
