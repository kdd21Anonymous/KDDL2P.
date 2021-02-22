import argparse
import time

import torch.nn.functional as F
from torch.nn import Linear

from datasets import *
from gcn_convNet import OurGCN2ConvNewData
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=int, default=0)
parser.add_argument('--hidden', type=int, default=0)
parser.add_argument('--dropout', type=int, default=2)
parser.add_argument('--normalize_features', type=bool, default=True)  # False
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--laynumber', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--H0alpha', type=int, default=5)  # 2
parser.add_argument('--theta', type=float, default=0.0)
parser.add_argument('--concat', type=str, default='concat')  # concat

parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--dataset', type=str, default='cornell')  # squirrel film chameleon     cornell   texas   wisconsin
parser.add_argument('--eachLayer', type=int, default=1)  # 1 is 1 layer, XX otherwise
parser.add_argument('--trainType', type=str, default='val')  # train / XXX
parser.add_argument('--gamma', type=str, default='parameter')  # metaNet  parameter / XXX
parser.add_argument('--taskType', type=str, default='full')  # full,semi
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--entropy', type=int, default=5)

args = parser.parse_args()
task_type = args.taskType  # full,semi
gamma = args.gamma
print('gamma is %s, traintype is %s' % (args.gamma, args.trainType))

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


set_seed(args.seed)
weightDecayL = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
args.weight_decay = weightDecayL[args.weight_decay]
hiddenL = [64, 128, 256]
args.hidden = hiddenL[args.hidden]
args.entropy = args.entropy / 5.0
args.dropout = args.dropout / 5.0
args.H0alpha = args.H0alpha * 0.2
# print('gamma is %s, traintype is %s'%(args.gamma,args.trainType))
print('dataset: %s' % args.dataset)
print('H0alpha: %s, entropy: %s, alpha: %s, weight_decay: %s, hidden: %s, dropout: %s'
      % (args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout))


def get_t(retain_score):
    tMatrix = 1
    for count in range(retain_score.shape[1]):
        if count == 0:
            tMatrix = torch.sigmoid(retain_score[:, count]).reshape(retain_score.shape[0], 1)
        else:
            if count == retain_score.shape[1] - 1:
                t = 1
            else:
                t = torch.sigmoid(retain_score[:, count])
            for i in range(count):
                t = (1 - torch.sigmoid(retain_score[:, i])) * t
            tMatrix = torch.cat((tMatrix, t.reshape(tMatrix.shape[0], 1)), 1)

    # index = torch.distributions.Categorical(tMatrix).sample().unsqueeze(1)
    # tSample = torch.zeros_like(tMatrix, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0).detach()

    ent_loss = 0.1 * torch.distributions.Categorical(tMatrix).entropy().mean()
    tMatrix = F.gumbel_softmax((tMatrix), tau=1, hard=False)
    # tMatrix = F.gumbel_softmax(torch.log(tMatrix), tau=1, hard=False)
    print(tMatrix.max(dim=-1))
    return tMatrix, ent_loss


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, gamma='none', eachLayer=True):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        # self.lins.append(Linear(hidden_channels, dataset.num_classes))
        # if args.concat == 'concat':
        #     self.lins.append(Linear(hidden_channels*2, dataset.num_classes))
        # else:
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        self.convsVal = torch.nn.ModuleList()
        self.dropout = dropout
        self.ParameterList = torch.nn.ParameterList()

        self.pai = torch.nn.Parameter(torch.randn(1, num_layers + 1))  # .to(torch.device('cuda'))
        # a = torch.Tensor(np.ones((1,num_layers+1)))
        # self.pai.data = a
        self.ParameterList.append(self.pai)
        # if args.concat == 'concat':
        #     self.MetaNet = Linear(hidden_channels * 2, 1)
        # else:
        self.MetaNet = Linear(hidden_channels, 1)
        self.convsVal.append(self.MetaNet)
        self.paramsVal = list(self.convsVal.parameters())

        for layer in range(num_layers):
            self.convs.append(
                OurGCN2ConvNewData(hidden_channels, 0.0, 0.0, layer + 1,
                                   shared_weights, normalize=False))

    def forward(self, x, adj_t, mask, type):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = F.dropout(self.lins[0](x).relu(), self.dropout, training=self.training)  #
        count = 0
        preds = []
        xMatrix = []
        loss_list = []
        # preds.append(x)
        # if args.concat == 'concat':
        #     preds.append(torch.cat((x, x_0), 1))
        # else:
        preds.append(x)
        # if args.concat == 'concat':
        #     xMatrix.append(torch.cat((x, args.H0alpha * x_0), 1))
        #     predsI = self.lins[1](torch.cat((x, args.H0alpha * x_0), 1)).log_softmax(dim=-1)
        # else:
        xMatrix.append(x)
        for conv in self.convs:
            count += 1
            x = F.dropout(x, self.dropout, training=self.training)
            conv.alpha = 0.0  # args.alpha#0.1
            conv.beta = 0.0
            edge_index = adj_t._indices()
            edge_weight = adj_t._values()
            x = conv(x, x_0, edge_index, edge_weight)
            # x = x.relu()
            # if args.concat == 'concat':
            #     preds.append(torch.cat((x, x_0), 1))
            # else:
            preds.append(x)
            # if args.concat == 'concat':
            #     xMatrix.append(torch.cat((x, args.H0alpha * x_0), 1))
            #     predsI = F.dropout(torch.cat((x, args.H0alpha * x_0), 1), self.dropout, training=self.training)
            #     predsI = self.lins[1](predsI).log_softmax(dim=-1)
            # else:
            xMatrix.append(x)
            # predsI = F.dropout(x, self.dropout, training=self.training)
            # predsI = self.lins[1](predsI).log_softmax(dim=-1)
            # loss = F.nll_loss(predsI[mask], data.y[mask],reduction="none")
            # loss_list.append(loss)
        xMatrix = torch.stack(xMatrix, dim=1)
        pps = torch.stack(preds, dim=1)  # n*(L+1)*k
        # pi = self.MetaNet(pps).squeeze()  # n*(L+1)
        # #pi = get_t(pi)
        # pi=torch.softmax(pi,dim=-1)
        retain_score = torch.exp(self.pai).unsqueeze(1)
        # retain_score = self.pai.unsqueeze(1)  # n*1*(L+1)
        x = torch.matmul(retain_score, xMatrix).squeeze()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1), 0.0

    #
    # def forward2(self, x, adj_t, mask,type):
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = x_0 = F.dropout(self.lins[0](x).relu(), self.dropout, training=self.training) #
    #     count = 0
    #     preds = []
    #     preds.append(x)
    #     loss_list = []
    #     predsI = self.lins[1](x).log_softmax(dim=-1)
    #     loss = F.nll_loss(predsI, data.y,reduction="none")
    #     loss_list.append(loss)
    #     for conv in self.convs:
    #         count += 1
    #         x = F.dropout(x, self.dropout, training=self.training)
    #         conv.alpha = args.alpha#0.1
    #         conv.beta =0.0
    #         x = conv(x, x_0, adj_t)
    #         x = x.relu()
    #         preds.append(x)
    #         predsI = F.dropout(x, self.dropout, training=self.training)
    #         predsI = self.lins[1](predsI).log_softmax(dim=-1)
    #         loss = F.nll_loss(predsI, data.y,reduction="none")
    #         loss_list.append(loss)
    #
    #
    #     pps = torch.stack(preds, dim=1)  # n*(L+1)*k
    #     pi = self.MetaNet(pps)  # n*(L+1)*1
    #     pi =pi.squeeze()  # n*(L+1)
    #     # tMatrix, retain_score= get_t(retain_score)
    #     pi=torch.softmax(pi,dim=-1)
    #
    #     loss_list=torch.stack(loss_list,dim=1)
    #     loss_label=torch.softmax(-loss_list,dim=1)
    #
    #     index = torch.distributions.Categorical(loss_label).sample().unsqueeze(1)
    #     retain_score = torch.zeros_like(loss_label, memory_format=torch.legacy_contiguous_format).scatter_(-1, index,
    #                                                                                                1.0).detach()
    #     if type=="test":
    #         index = torch.distributions.Categorical(pi).sample().unsqueeze(1)
    #         retain_score = torch.zeros_like(pi, memory_format=torch.legacy_contiguous_format).scatter_(-1,
    #                                                                                                            index,
    #                                                                                                            1.0).detach()
    #
    #     retain_score = retain_score.unsqueeze(1)  # n*1*(L+1)
    #     x = torch.matmul(retain_score, pps).squeeze()
    #     x = F.dropout(x, self.dropout, training=self.training)
    #     x = self.lins[1](x)
    #
    #
    #     klLoss = torch.mean(-loss_label[mask].detach()*torch.log(pi[mask]))
    #
    #
    #     return x.log_softmax(dim=-1), klLoss


#
#
# device = torch.device('cuda')
# model = Net(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta,
#             shared_weights=True, dropout=args.dropout,gamma = gamma,eachLayer=args.eachLayer).to(device)
# data = data.to(device)
#
#
# optimizer = torch.optim.Adam([
#     dict(params=model.convs.parameters(), weight_decay=5e-4),
#     dict(params=model.lins.parameters(), weight_decay=5e-4),
#     dict(params=model.convsVal.parameters(), weight_decay=5e-4),
# ], lr=0.01)


def train(model, optimizer, data, adj):
    model.train()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    optimizer.zero_grad()
    if args.trainType != 'train':  # Val not train
        if gamma == 'parameter':
            for param in model.ParameterList:
                param.requires_grad = False
            for param in model.paramsVal:
                param.requires_grad = False
    out, klLoss = model(data.x, adj, data.train_mask, "train")
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


def Val_train(model, optimizer, data, adj):
    model.train()
    optimizer.zero_grad()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    # if gamma == 'parameter':
    for param in model.ParameterList:
        param.requires_grad = True
    for param in model.paramsVal:
        param.requires_grad = True

    out, klLoss = model(data.x, adj, data.val_mask, "val")

    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, optimizer, data, device, adj):
    model.eval()
    out, _ = model(data.x, adj, data.test_mask, "test")
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs_loss.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        accs_loss.append(F.nll_loss(out[mask], data.y[mask].to(device)))
    return accs_loss


@torch.no_grad()
def testGroup(model, optimizer, data, adj, dataset, device):
    model.eval()
    out, ent_loss = model(data.x, adj, data.test_mask, "test")
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []

    groupL = getDegreeGroup(dataset)
    for group in groupL:
        if len(group) != 0:
            FalseTensor = torch.tensor(dataset.data.train_mask)
            for i in range(dataset.data.num_nodes):
                FalseTensor[i] = False
            groupmask = torch.tensor(FalseTensor).index_fill_(0, torch.tensor(group).to(device), True)
            # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs_loss.append(int((pred[groupmask] == data.y[groupmask]).sum()) / int(groupmask.sum()))
            accs_loss.append(F.nll_loss(out[groupmask], data.y[groupmask].to(device)))
    return accs_loss


def getDegreeGroup(dataset):
    edge_index = dataset.data.edge_index
    e0 = edge_index[0]
    dict = {}
    for key in e0:
        dict[int(key)] = dict.get(int(key), 0) + 1

    totalList = 0
    groupL = []
    i = 0
    while totalList < len(dict.keys()):
        group = []
        lower = 2 ** i
        higher = 2 ** (i + 1)
        for key in dict.keys():
            if higher > dict.get(int(key), 0) >= lower:
                group.append(int(key))
        groupL.append(group)
        totalList += len(group)
        # print(group)
        # print(totalList)
        i += 1
    return groupL


def allTrain(data, adj):
    device = torch.device('cuda')
    model = Net(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta,
                shared_weights=True, dropout=args.dropout, gamma=gamma, eachLayer=args.eachLayer).to(device)
    # model = Net1(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta,
    #             shared_weights=True, dropout=args.dropout, gamma=gamma, eachLayer=args.eachLayer).to(device)

    # model = Net1(dataset, hidden_channels=64, num_layers=8, alpha=0.2, theta=1.5,
    #     shared_weights=True, dropout=0.6).to(device)
    data = data.to(device)
    # getDegreeGroup(dataset)
    # groupL = getDegreeGroup(dataset)
    # groupNum = []
    # for i in groupL:
    #     groupNum.append(len(i))
    # print(groupNum)
    # optimizer = torch.optim.Adam([
    #     dict(params=model.convs.parameters(), weight_decay=args.weight_decay),
    #     dict(params=model.lins.parameters(), weight_decay=args.weight_decay),
    #     dict(params=model.ParameterList,weight_decay=args.weight_decay),
    #     #dict(params=model.theta, weight_decay=0.01),
    #     dict(params=model.convsVal.parameters(), weight_decay=args.weight_decay),
    # ], lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    # convsVal, ParameterList, lins, convs
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    best_test = 0
    bigest_test = 0
    bigest_epoch = 0
    patienceEpoch = 0
    patienceTest = 0
    trainingLossL = []
    valLossL = []
    for epoch in range(args.epochs):
        loss_tra = train(model, optimizer, data, adj)
        if args.trainType != 'train':
            _ = Val_train(model, optimizer, data, adj)
        train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test(model, optimizer, data, device, adj)
        accs_loss = testGroup(model, optimizer, data, adj, dataset, device)
        import numpy as np

        listN = []
        for i in range(int(len(accs_loss) / 2)):
            listN.append(accs_loss[i * 2])
        print(listN)
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(train_loss),
                  'acc:{:.2f}'.format(train_acc * 100),
                  '| val',
                  'loss:{:.3f}'.format(val_loss),
                  'acc:{:.2f}'.format(val_acc * 100),
                  'acc_test:{:.2f}'.format(tmp_test_acc),
                  'best_acc_test:{:.2f}'.format(best_test * 100))
        trainingLossL.append(float(train_loss))
        valLossL.append(float(val_loss))
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            acc = val_acc
            # if best_test <tmp_test_acc:
            if bigest_test < tmp_test_acc:
                bigest_test = tmp_test_acc
                bigest_epoch = epoch
            best_test = tmp_test_acc
            #
            bad_counter = 0
            torch.save(model.state_dict(), 'params%s.pkl' % args.laynumber)

            # if tmp_test_acc > 0.834:
            # if epoch >50:
            # out,_ = model(data.x, adj, data.test_mask,"test")
            # outputT = (model.pai.cpu().detach().numpy())
            # newarray = np.zeros((len(groupL),outputT.shape[1]))
            # for i in range(len(groupL)):
            #     temp = outputT[groupL[i],:]
            #     newarray[i,:] = np.mean(temp, axis=0)

            # checkpt_file = '%sresultfree%s__' % (
            # args.taskType, args.laynumber) + args.dataset + '_ConcatbestTest_%s' % tmp_test_acc \
            #                + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
            #                    args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
            #                    args.dropout) + '.csv'
            # np.savetxt(checkpt_file, outputT, delimiter=',')
            # checkpt_file = '%sresultfreeTrue%s__' % (
            #     args.taskType, args.laynumber) + args.dataset + 'Epoch_%s' % epoch + '.csv'
            # np.savetxt(checkpt_file, outputT, delimiter=',')


        else:
            bad_counter += 1

        if bad_counter == args.patience:
            patienceEpoch = epoch
            patienceTest = tmp_test_acc
            break

    # if args.test:
    #     acc = test()[1]
    newarray = np.zeros((2, len(trainingLossL)))
    newarray[0, :] = trainingLossL
    newarray[1, :] = valLossL
    # checkpt_file = '%sresultfree%s__' % (
    # args.taskType, args.trainType) + args.dataset + '_ConcatbestTest_%s' % tmp_test_acc \
    #                + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
    #                    args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
    #                    args.dropout) + '.csv'
    checkpt_file = '%sresultfree%s__' % (args.taskType, args.trainType) + args.dataset + '_%s' % tmp_test_acc \
                   + '_layer%s' % (args.laynumber) + '.csv'
    np.savetxt(checkpt_file, newarray, delimiter=',')
    print("Train cost: {:.4f}s".format(time.time() - t_total))
    try:
        print('Max Load {}th epoch'.format(bigest_epoch))
        print("Max Test", "acc.:{:.1f}".format(bigest_test * 100))
    except:
        bigest_epoch = 0
        bigest_test = 0
    print('Load {}th epoch'.format(best_epoch))
    print("Test", "acc.:{:.1f}".format(best_test * 100))
    print('patience Load {}th epoch'.format(patienceEpoch))
    print("patience Test", "acc.:{:.1f}".format(patienceTest * 100))
    model.load_state_dict(torch.load('params%s.pkl' % args.laynumber))
    train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test(model, optimizer, data, device)
    accs_loss = testGroup(model, optimizer, data, adj, dataset, device)
    import numpy as np
    listN = []
    for i in range(int(len(accs_loss) / 2)):
        listN.append(accs_loss[i * 2])
    print(listN)
    np.savetxt('freefile%s%s.csv' % (args.laynumber, args.dataset), np.array(listN), delimiter=',')
    return (
        bigest_epoch, bigest_test, best_epoch, best_test, patienceEpoch, patienceTest, t_total, model, trainingLossL,
        valLossL)


# chameleon     cornell   texas   wisconsin
# if args.dataset == "chameleon" or args.dataset == "cornell" or args.dataset == "texas" or args.dataset == "wisconsin":
from process import *

datastr = args.dataset
acc_list = []
for i in range(10):
    splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr, splitstr)
    adj = adj.to(torch.device('cuda:0'))
    dataset = get_planetoid_dataset('cora', "True")
    dataset.data.train_mask = idx_train
    dataset.data.test_mask = idx_test
    dataset.data.val_mask = idx_val
    dataset.data.edge_index = adj._indices()
    # dataset.data.edge_index = torch.LongTensor(np.concatenate([adj.row.reshape(1, -1), adj.col.reshape(1, -1)], axis=0))
    # dataset.data.edge_index = torch.long(np.concatenate([adj.row.reshape(1, -1), adj.col.reshape(1, -1)], axis=0)) #adj._indices()
    dataset.data.x = features
    dataset.data.y = labels
    # dataset.data.edge_attr = adj._values()
    data = dataset.data

    # dataset.num_features = data.num_features
    # dataset.num_classes = labels.max().numpy()+1
    bigest_epoch, bigest_test, best_epoch, best_test, patienceEpoch, patienceTest, t_total, model, trainingLossL, valLossL = allTrain(
        data, adj)

    acc_list.append(best_test)
    print(i, ": {:.2f}".format(acc_list[-1]))
    print(acc_list)
print("Train cost: {:.4f}s".format(time.time() - t_total))
print(acc_list)
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
# newarray = np.zeros((2,len(trainingLossL)))
# newarray[0,:] = trainingLossL
# newarray[1,:] = valLossL
# checkpt_file = '%sresultfree%s__' % (args.taskType,args.trainType) + args.dataset + '_ConcatbestTest_%s' % np.mean(acc_list) \
#                + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                    args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                    args.dropout) + '.csv'
# np.savetxt(checkpt_file, newarray, delimiter=',')
#
# checkpt_file = 'fullresult/%sfullresultpiklloss__' % args.taskType + args.dataset + '_ConcatbestTest_%s' % np.mean(acc_list) + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.txt'
#
# np.savetxt(checkpt_file,acc_list,delimiter=' ')

# t_total = time.time()
# bad_counter = 0
# best = 999999999
# best_epoch = 0
# acc = 0
# best_test=0
# bigest_test=0
# patienceEpoch = 0
# patienceTest = 0
#
# for epoch in range(args.epochs):
#     loss_tra = train(model,optimizer,data,adj)
#     if args.trainType != 'train':
#         _ = Val_train(model,optimizer,data,adj)
#     train_acc, train_loss ,val_acc, val_loss,tmp_test_acc,test_loss = test()
#     if(epoch+1)%1 == 0:
#         print('Epoch:{:04d}'.format(epoch+1),
#             'train',
#             'loss:{:.3f}'.format(train_loss),
#             'acc:{:.2f}'.format(train_acc*100),
#             '| val',
#             'loss:{:.3f}'.format(val_loss),
#             'acc:{:.2f}'.format(val_acc*100),
#             'acc_test:{:.2f}'.format(tmp_test_acc),
#         'best_acc_test:{:.2f}'.format(best_test*100))
#
#     if val_loss < best:
#         best = val_loss
#         best_epoch = epoch
#         acc = val_acc
#         #if best_test <tmp_test_acc:
#         if bigest_test <tmp_test_acc:
#             bigest_test = tmp_test_acc
#             bigest_epoch = epoch
#         best_test=tmp_test_acc
#         #
#         bad_counter = 0
#     else:
#         bad_counter += 1
#
#     if bad_counter == args.patience:
#         patienceEpoch =epoch
#         patienceTest = tmp_test_acc
#
#
#
# # if args.test:
# #     acc = test()[1]
#
# print("Train cost: {:.4f}s".format(time.time() - t_total))
# try:
#     print('Max Load {}th epoch'.format(bigest_epoch))
#     print("Max Test","acc.:{:.1f}".format(bigest_test*100))
# except:
#     1
# print('Load {}th epoch'.format(best_epoch))
# print("Test","acc.:{:.1f}".format(best_test*100))
# print('patience Load {}th epoch'.format(patienceEpoch))
# print("patience Test","acc.:{:.1f}".format(patienceTest*100))


#
# if task_type == 'full':
#     if args.dataset == 'cora':
#         if best_test > 0.8:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.8:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.8:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'CiteSeer':
#         if best_test > 0.79:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.79:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.79:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'PubMed':
#         if best_test > 0.9:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.9:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.9:
#             checkpt_file = 'fullresult/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
# else:
#     if args.dataset == 'cora':
#         if best_test > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'CiteSeer':
#         if best_test > 0.7:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.7:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.7:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'PubMed':
#         if best_test > 0.77:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.77:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.77:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'cs':
#         if best_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'physics':
#         if best_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'computers':
#         if best_test > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.8:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
#     if args.dataset == 'photo':
#         if best_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif patienceTest > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#         elif bigest_test > 0.9:
#             checkpt_file = 'result/' + args.dataset + '/piKLbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                 args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                 args.dropout) + '.pt'
#             cudaid = "cuda:" + str(args.dev)
#             print(cudaid, checkpt_file)
#             torch.save(model.state_dict(), checkpt_file)
#
