# from tkinter.messagebox import NO
import numpy as np
from sklearn.metrics import precision_recall_fscore_support,f1_score
import os

import warnings
warnings.filterwarnings("ignore")



def eval_m3ed(results, truths, choice_modality=None, test=None):

    test_preds = results.cpu().detach().numpy()   #（utterance总个数,7）
    test_truth = truths.cpu().detach().numpy()  #（utterance总个数）
    predicted_label = []
    true_label = []
    #预测总的
    for i in range(test_preds.shape[0]):
        predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
        true_label.append(test_truth[i])
    if test:
        load_pre_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))
        save_path = os.path.join(load_pre_path, 'save/pre_true_label_{}.txt'.format(choice_modality))
        fout_p = open(save_path, 'w')
        j=0
        for i in range(len(predicted_label)):
            pred_utt = str(predicted_label[i])
            true_utt = str(true_label[i])
            if predicted_label[i] == true_label[i]:
                j+=1
            fout_p.write(pred_utt + ' ' + true_utt + '\n')

        print(f"Saved labels at /save/pre_true_labels.txt!")
        print('共'+ str(len(predicted_label)) + '个utterance,' + '对了' + str(j) + '个')
        fout_p.close()
        print('Weighted FScore: \n ', f1_score(true_label, predicted_label, average='macro'))
    else:
        macro_f1 = f1_score(true_label, predicted_label, average='macro')
        return macro_f1



