import torch
import torch.nn as nn

class LSG(nn.Module):
    def __init__(self, network, backbone, text_features,
                       class_num=100, edge_ratio=0.002, w_ti=1.0, w_ii=1.0, pretrained=True):
        super(LSG, self).__init__()
        self.class_num = class_num
        self.backbone = backbone
        self.pretrained = pretrained
        self.edge_ratio = edge_ratio
        self.text_features = text_features
        self.w_ti = w_ti
        self.w_ii = w_ii
        print('\n****************** Constructing LSG ******************')
        print("Adjacency matrix weight: w_ti:{}, w_ii:{}".format(self.w_ti, self.w_ii))

        n_classes, n_prompts, _ = text_features.size()
        text_labels = []
        for i in range(n_classes):
            text_labels += [i for j in range(n_prompts)]
        text_labels = torch.tensor(text_labels)
        self.text_labels = text_labels

        # create the encoders
        # print(network)
        self.encoder = network(text_features=text_features, text_labels=text_labels, edge_ratio=self.edge_ratio)

        self.load_pretrained(network)

    def load_pretrained_gcn(self, model_path):
        # load pretrained gcn
        gcn = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict_new = {}
        for k, v in gcn.items():
            k = k.replace("gnn", "encoder.gnn")
            state_dict_new[k] = v
        self.load_state_dict(state_dict_new,strict=False)

    def forward(self, img, label):
        """
        Input:
            img: a batch of images
            label: corresponding labels
        Output:
            graph output, graph embedding, img projected embedding, image embedding
        """
        c, g, h, f = self.encoder(img, label, self.w_ti, self.w_ii) 
        return c, g, h, f

    def load_pretrained(self, network):
        if 'resnet' in self.backbone:
            enc = network(text_features=self.text_features,text_labels=self.text_labels,pretrained=self.pretrained, edge_ratio=self.edge_ratio)
            self.encoder = enc
        else:
            raise NotImplementedError

    def inference(self, img):
        feat = self.encoder.inference(img)
        return feat

    def gcn_inference(self, img, pred):
        output = self.encoder.gcn_inference(img, pred)
        return output

