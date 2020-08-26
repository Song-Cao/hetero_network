import torch
import numpy as np
from sklearn.metrics import f1_score

from utils import load_data, EarlyStopping

def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()

    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    recall = 1.0*(prediction * labels == 1).sum() / (labels == 1).sum()
    precision = 1.0*(prediction * labels == 1).sum() / (prediction == 1).sum()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1, prediction, recall, precision

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1, prediction, recall, precision = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, prediction, recall, precision

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_data(args['dataset'])
    # print(labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features = features.to(args['device'])
    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    if args['hetero']:
        from model_hetero import HAN
        model = HAN(meta_paths=[['pa', 'ap'], ['pf', 'fp']],
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = g.to(args['device'])
    else:
        from model import HAN
        model = HAN(num_meta_paths=len(g),
                    in_size=features.shape[1],
                    hidden_size=args['hidden_units'],
                    out_size=num_classes,
                    num_heads=args['num_heads'],
                    dropout=args['dropout']).to(args['device'])
        g = [graph.to(args['device']) for graph in g]

    stopper = EarlyStopping(patience=args['patience'])
    # loss_fcn = torch.nn.CrossEntropyLoss()
    class_weights = torch.FloatTensor(args['weight']).cuda()
    loss_fcn = torch.nn.CrossEntropyLoss(weight=class_weights) # I think the weight mimics parameter in network propagation
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1, train_prediction, train_recall, train_precision = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1, val_prediction, val_recall, val_precision = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | Train Recall {:.4f} | Train Precision {:.4f} | Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f} | Val Recall {:.4f} | Val Precision {:.4f}'.format(
            epoch + 1, loss.item(), train_micro_f1, train_macro_f1, train_recall, train_precision, val_loss.item(), val_micro_f1, val_macro_f1, val_recall, val_precision))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1, test_prediction, test_recall, test_precision = evaluate(model, g, features, labels, test_mask, loss_fcn)

    # output train, val, test prediction in the same array
    all_prediction = np.zeros(len(labels),)
    all_prediction[train_mask.cpu()] = train_prediction # does train mask need to be convert back to cpu
    all_prediction[val_mask.cpu()] = val_prediction
    all_prediction[test_mask.cpu()] = test_prediction
    np.save(args['ds'] + '_' + args['runid'] + '_pred', all_prediction)

    # output attention values
    semantic_attention = model.layers[0].semantic_attention.beta.cpu()
    np.save(args['ds'] + '_' + args['runid'] + '_semantic_attn', semantic_attention)

    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f} | Test Recall f1 {:.4f}  | Test Precision {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1, test_recall, test_precision))

if __name__ == '__main__':
    import argparse

    from utils import setup

    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset')
    parser.add_argument('--ds', help='customized data set')
    parser.add_argument('--weight', nargs='+', default=[1.0,1.0], type=float, help='a weight passed to the class 0 (major class). It is in order to fight with the class imbalance. It should be similar to parameters in network propagation')
    parser.add_argument('--runid', required=True, help='a string to distinguish the identity of experiment')
    args = parser.parse_args().__dict__
    args = setup(args)

    main(args)
