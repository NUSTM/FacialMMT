import numpy as np
from sklearn.metrics import f1_score

# def eval_affwild(preds, label_orig):
#     val_preds = preds.cpu().detach().numpy()
#     val_true = label_orig.cpu().detach().numpy() 
#     predicted_label = []
#     true_label = []
#     for i in range(val_preds.shape[0]):
#         predicted_label.append(np.argmax(val_preds[i,:],axis=0) ) #
#         true_label.append(val_true[i])
#     macro_av_f1 = f1_score(true_label, predicted_label, average='macro')
#     return macro_av_f1


def eval_meld(results, truths, test=False):
    test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
    test_truth = truths.cpu().detach().numpy()  #（num_utterance）
    predicted_label = []
    true_label = []
    for i in range(test_preds.shape[0]):
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
        true_label.append(test_truth[i])
    wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
    if test:
        f1_each_label = f1_score(true_label, predicted_label, average=None)
        print('**TEST** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
    return wg_av_f1

