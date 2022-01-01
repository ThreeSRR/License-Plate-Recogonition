import numpy as np
from utils import sparse_tuple_for_ctc, decode, CHARS


def eval(lprnet, STN, criterion, T_length, dataloader, dataset, device):
    
    lprnet = lprnet.to(device)
    STN = STN.to(device)
    TP = 0
    total_loss = []

    for imgs, labels, lengths in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = STN(imgs)
        logits = lprnet(transfer) 
        
        log_probs = logits.permute(2, 0, 1)
        log_probs = log_probs.log_softmax(2).requires_grad_()       
        input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
        loss = criterion(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)

        total_loss.append(loss.item())

        preds = logits.cpu().detach().numpy()
        _, pred_labels = decode(preds, CHARS)

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start:start+length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                TP += 1
            
    ACC = TP / len(dataset) 
    
    return sum(total_loss)/len(total_loss), ACC
