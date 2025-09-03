import torch
from utils import * 
from model import LayerNormNet
from distance_map import *
from evaluate import *
import pandas as pd
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def infer_maxsep(train_data, test_data, report_metrics = False, 
                model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/datasets/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/datasets/' + test_data + '.csv')
    # load checkpoints
    model = LayerNormNet(512, 128, device, dtype)
    try:
        checkpoint = torch.load('./data/model/'+ model_name +'.pth', map_location=device)
    except FileNotFoundError as error:
        raise Exception('No model found!')     
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    emb_train,_ = model(esm_embedding(ec_id_dict_train, device, dtype))
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results")
    out_filename = "./results/" +  test_data
    write_max_sep_choices(eval_df, out_filename)
    if report_metrics:
        pred_label = get_pred_labels(out_filename, pred_type='_maxsep')
        pred_probs = get_pred_probs(out_filename, pred_type='_maxsep')
        true_label, all_label = get_true_labels('./data/datasets/' + test_data)
        pre, rec, f1, roc, acc = get_eval_metrics(
            pred_label, pred_probs, true_label, all_label)
        print("############ EC calling results using maximum separation ############")
        print('-' * 75)
        print(f'>>> total samples: {len(true_label)} | total ec: {len(all_label)} \n'
            f'>>> precision: {pre:.4} | recall: {rec:.4}'
            f'| F1: {f1:.4} | AUC: {roc:.4} ')
        print('-' * 75)
        
def use_maxsep_pred_ec_number(train_data, test_data, model_name=None):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    id_ec_train, ec_id_dict_train = get_ec_id_dict('./data/datasets/' + train_data + '.csv')
    id_ec_test, _ = get_ec_id_dict('./data/datasets/' + test_data + '.csv')
    # load checkpoints
    model = LayerNormNet(512, 128, device, dtype)
    try:
        checkpoint = torch.load('./model/train_model/'+ model_name +'.pth', map_location=device)
    except FileNotFoundError as error:
        raise Exception('No model found!')     
    model.load_state_dict(checkpoint)
    model.eval()
    # load precomputed EC cluster center embeddings if possible
    # emb_train,_ = model(esm_embedding(ec_id_dict_train, device, dtype))
    with open('./data/datasets_process/train_dataset_model_embedding/' + train_data + '_model_embedding.pkl', 'rb') as f:
        emb_train = pickle.load(f)
        
    emb_test = model_embedding_test(id_ec_test, model, device, dtype)
    eval_dist = get_dist_map_test(emb_train, emb_test, ec_id_dict_train, id_ec_test, device, dtype)
    seed_everything()
    eval_df = pd.DataFrame.from_dict(eval_dist)
    ensure_dirs("./results/tmp")
    out_filename = "./results/tmp/" +  test_data 
    write_max_sep_choices(eval_df, out_filename)
    print('####################################')
    print('pred_ec_number_save: ', out_filename+"_maxpsep.csv")

        
        
