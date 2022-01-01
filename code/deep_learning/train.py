import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter

from models import LPRNet, STNet
from utils import sparse_tuple_for_ctc, CHARS
from utils import LPRDataset, collate_fn
from utils import eval, decode


def main(train_data, val_data, img_size, batch_size, num_epochs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    lprnet = LPRNet(class_num=len(CHARS))
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('weights/LPRNet_model_Init.pth', map_location=lambda storage, loc: storage))
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('weights/STN_model_Init.pth', map_location=lambda storage, loc: storage))

    train_dataset = LPRDataset([train_data], img_size)
    val_dataset = LPRDataset([val_data], img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)


    lr = 0.001
    gamma=0.1
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    

    optimizer = torch.optim.Adam([{'params': STN.parameters(), 'weight_decay': 2e-5},
                                  {'params': lprnet.parameters()}], lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean')

    save_dir = os.path.join('./output', time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    total_iters = 0
    best_acc = 0
    best_iter = 0
    T_length = 18
    best_lprnet = lprnet
    best_stn = STN
    
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'Tensorboard_Results'))

    for epoch in trange(num_epochs):

        lprnet.train()
        STN.train()

        for imgs, labels, lengths in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):

                transfer = STN(imgs)

                logits = lprnet(transfer)
                log_probs = logits.permute(2, 0, 1)
                log_probs = log_probs.log_softmax(2).requires_grad_()       
                input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
                loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
                
                loss.backward()
                optimizer.step()
                
                total_iters += 1

                writer.add_scalar('train_loss_logs', loss.item(), total_iters)
                
                if total_iters % 100 == 0:
                                
                    preds = logits.cpu().detach().numpy()
                    _, pred_labels = decode(preds, CHARS)
                    total = preds.shape[0]
                    start = 0
                    TP = 0
                    for i, length in enumerate(lengths):
                        label = labels[start:start+length]
                        start += length
                        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                            TP += 1

                    for p in optimizer.param_groups:
                        lr = p['lr']

                    print("Epoch %d/%d, Iters: %d, learning rate: %f, loss: %.4f, train_accuracy: %.4f"
                          % (epoch+1, num_epochs, total_iters, lr, loss.item(), TP/total))

            
            if total_iters % 800 == 0:
                
                lprnet.eval()
                STN.eval()

                train_loss, train_acc = eval(lprnet, STN, ctc_loss, T_length, train_loader, train_dataset, device)
                test_loss, acc = eval(lprnet, STN, ctc_loss, T_length, val_loader, val_dataset, device)

                writer.add_scalar('train_acc', train_acc, total_iters)
                writer.add_scalar('test_acc', acc, total_iters)
                writer.add_scalar('test_loss', test_loss, total_iters)

                            
                if best_acc <= acc:
                    best_acc = acc
                    best_iter = total_iters

                    best_lprnet = lprnet
                    best_stn = STN
                
                lprnet.train()
                STN.train()

                print("Epoch %d/%d, Iters: %d, test_loss: %.4f, test_accuracy: %.4f, current_best_acc: %.4f"
                          % (epoch+1, num_epochs, total_iters, test_loss, acc, best_acc))

        scheduler.step()

    print('Best Accuracy: %.4f in iters: %d' % (best_acc, best_iter))
    torch.save(best_lprnet.state_dict(), os.path.join(save_dir, 'best_LPRNet_model.pth'))
    torch.save(best_stn.state_dict(),  os.path.join(save_dir, 'best_STN_model.pth'))

    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', default=(94, 24), help='input image size')
    parser.add_argument('--train_data', default="./data/train", help='dir path of training data')
    parser.add_argument('--val_data', default="./data/validation", help='dir path of validation data')
    parser.add_argument('--num_epochs', default=60, type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    args = parser.parse_args()

    img_size = args.img_size
    train_data = args.train_data
    val_data = args.val_data
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    main(train_data, val_data, img_size, batch_size, num_epochs)
