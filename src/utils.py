import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle

from pathlib import Path

import torch.nn as nn
import torchvision
from tqdm import tqdm
import pandas as pd
import esm
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

from distance_map import get_dist_map


def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            for ec in rows[1].split(';'):
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][36]
    return a


def load_esm(lookup):
    esm = format_esm(torch.load('./data/esm_data/' + lookup + '.pt'))
    return esm.unsqueeze(0)


def esm_embedding(ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(id_ec_test, model, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb,_ = model(esm_emb)
    return model_emb

def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')

def csv_to_fasta_test_dataset(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[1] + '\n')
            
def fasta_to_csv(fasta_file, csv_file):
    records = []
    with open(fasta_file, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            records.append({
                "Entry": record.id,          # 序列ID
                "Sequence": str(record.seq)  # 序列
            })
    
    # 转换为 DataFrame 并保存为 CSV
    df = pd.DataFrame(records)
    df.to_csv(csv_file, sep='\t',index=False)
    print(f" fasta_to_csv save: {csv_file}")
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm2_embedding_36_mean(fasta_name):
    esm_script = "./src/extract.py"
    esm_out = "./data/datasets_process/esm2_rep"
    esm_type = "esm2_t36_3B_UR50D"
    fasta_name = "./data/datasets/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)

def compute_esm_distance(train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('.=.//data/datasets/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open('./data/distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(esm_emb, open('./data/distance_map/' + train_file + '_esm.pkl', 'wb'))
    
def prepare_infer_fasta(fasta_name):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open('./data/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('./data/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name) :
    csv_file = open('./data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/' + fasta_name + '.fasta','w')
    single_id = set(single_id)
    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(10):
                seq = rows[2].strip()
                mu, sigma = .10, .02 # mean and standard deviation
                s = np.random.normal(mu, sigma, 1)
                mut_rate = s[0]
                times = math.ceil(len(seq) * mut_rate)
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                seq = seq.replace('*', '<mask>')
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def mutate_single_seq_ECs(train_file):
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv')
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
    single_id = set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'):
                single_id.add(id)
                break
    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    mask_sequences(single_id, train_file, train_file+'_single_seq_ECs')
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name


def create_ec_onehot_label(ec_anchor_pos_neg,label_to_idx):
    # 整理结果的容器
    anchor_one_hots = []
    pos_one_hots = []
    neg_one_hots = []
    
    for item in ec_anchor_pos_neg:
        anchor_one_hot = one_hot_encode(item['anchor_ec'], label_to_idx)
        pos_one_hot = one_hot_encode(item['pos_ec'], label_to_idx)
        neg_one_hot = one_hot_encode(item['neg_ec'], label_to_idx)

        # 将结果添加到对应的容器中
        anchor_one_hots.append(anchor_one_hot)
        pos_one_hots.append(pos_one_hot)
        neg_one_hots.append(neg_one_hot)
        
    anchor_one_hots_tensor = torch.stack(anchor_one_hots)
    pos_one_hots_tensor = torch.stack(pos_one_hots)
    neg_one_hots_tensor = torch.stack(neg_one_hots)
    
    return anchor_one_hots_tensor, pos_one_hots_tensor, neg_one_hots_tensor
    # return anchor_one_hots_tensor

   
# 创建一个 one-hot 编码函数
def one_hot_encode(ec_list, label_to_idx):
    one_hot = torch.zeros(( len(label_to_idx)))  # 创建零张量
    for i, ec in enumerate(ec_list):
        if ec in label_to_idx:  # 如果 EC 存在于标签集合中
            one_hot[label_to_idx[ec]] = 1  # 设置对应位置为 1
    return one_hot



def hierarchical_similarity(ec1, ec2, weights=None):
    """
    计算两个EC编号的层次相似性，为每一层定义自定义权重，并确保权重和为1
    :param ec1: 第一个EC编号，例如 "1.1.1.95"
    :param ec2: 第二个EC编号，例如 "1.1.1.399"
    :param weights: 每一层的权重列表，例如 [0.4, 0.3, 0.2, 0.1]
    :return: 相似性得分（0到1之间）
    """
    # 默认权重（如果未提供）
    if weights is None:
        # weights = [0.1, 0.2, 0.3, 0.4]  # 四层权重，总和为1
        weights = [0.4, 0.3, 0.2, 0.1]  # 四层权重，总和为1
        #  weights = [0.25, 0.25, 0.25, 0.25]
    
    # 归一化权重，确保总和为51
    weight_sum = sum(weights)
    if weight_sum != 1.0:
        weights = [w / weight_sum for w in weights]
    
    parts1 = ec1.split('.')
    parts2 = ec2.split('.')
    common_level = 0
    
    for i, (p1, p2) in enumerate(zip(parts1, parts2)):
        if i >= len(weights):
            break  # 如果层数超过权重列表长度，则停止
        if p1 != p2:
            return common_level  # 如果当前层不匹配，直接返回当前的相似性得分
        common_level += weights[i]
    
    return common_level


def compute_similarity(ec_list1, ec_list2, similarity_mode="max"):
    """
    计算两组EC编号之间的相似性（支持最大值或平均值）
    :param ec_list1: 第一组EC编号列表，例如 ['1.1.1.95', '1.1.1.399']
    :param ec_list2: 第二组EC编号列表，例如 ['1.1.1.95', '1.1.1.37']
    :param similarity_mode: 相似性计算模式，"max" 或 "mean"
    :return: 相似性得分
    """
    similarities = []
    for ec1 in ec_list1:
        for ec2 in ec_list2:
            similarities.append(hierarchical_similarity(ec1, ec2))
    
    if similarity_mode == "max":
        return max(similarities)
    elif similarity_mode == "mean":
        return sum(similarities) / len(similarities)
    else:
        raise ValueError(f"Unsupported similarity_mode: {similarity_mode}. Use 'max' or 'mean'.")
       
def dynamic_margin_multilabel(sim_pos, sim_neg, base_margin=1.0, min_margin=0.1, max_margin=20.0):
    """
    动态调整的边界值，并添加上下限
    """
    # margin = base_margin * (1 - sim_pos + sim_neg)
    margin = base_margin * (2 - sim_pos + sim_neg)
    return torch.clamp(margin, min=min_margin, max=max_margin)


def compute_contact_map(sequence, model, batch_converter, threshold=8.0):
    """
    计算蛋白质序列的 Contact Map。
    
    参数:
    - sequence (str): 蛋白质序列。
    - model (torch.nn.Module): 预加载的 ESM 模型。
    - batch_converter (callable): 用于将序列转换为 ESM 模型的输入。
    - threshold (float): 生成二值化 Contact Map 的距离阈值。
    
    返回:
    - contact_map (ndarray): 二值化接触图 (N x N)。
    - raw_contacts (ndarray): 原始的接触分数 (N x N)。
    """
    # 转换序列为模型输入
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [("protein", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)  # 将输入转移到 GPU
    
    # 使用模型提取 pairwise residue representations
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    # 提取 contact probabilities
    raw_contacts = results["contacts"][0].cpu().numpy()

    # 转换为二值化接触图
    contact_map = (raw_contacts > (threshold / 100.0)).astype(int)

    return contact_map, raw_contacts




def process_fasta_and_save_contact_maps(fasta_name, model_name="esm2_t33_650M_UR50D", threshold=8.0):
    """
    处理 FASTA 文件中的序列，计算接触图，并保存到指定目录。
    
    参数:
    - fasta_file (str): 输入的 FASTA 文件路径。
    - output_dir (str): 输出接触图保存的目录。
    - model_name (str): 使用的 ESM 模型名称。
    - threshold (float): 接触图的阈值。
    """
    
    fasta_file = "./data/datasets/" + fasta_name + ".fasta"
    
    device = 'cuda:0'
    # 加载 ESM2 模型
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()  # 设置为评估模式
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()


    #esm2_t33_650M_UR50D
    model, alphabet =   esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 确保输出目录存在
 
    contactmap_8a_path = './data/datasets_process/contactmap_rep/contactmap_8a'
    esm2_contactmap_path = './data/datasets_process/contactmap_rep/esm2_contactmap'
    os.makedirs(contactmap_8a_path, exist_ok=True)
    os.makedirs(esm2_contactmap_path, exist_ok=True)
    

    # 遍历 FASTA 文件中的序列
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)

        # print(f"Processing sequence {seq_id}...")

        # 计算 Contact Map
        contact_map, raw_contacts = compute_contact_map(
            sequence, model, batch_converter, threshold
        )

        # 保存 Contact Map 和原始概率矩阵
        
        
        contact_map_path = os.path.join(contactmap_8a_path, f"{seq_id}.npy")
        raw_contacts_path = os.path.join(esm2_contactmap_path , f"{seq_id}.npy")

        np.save(contact_map_path, contact_map)
        np.save(raw_contacts_path, raw_contacts)

        # print(f"Saved Contact Map for {seq_id} to {contact_map_path}")
        
        # 清除显存
        torch.cuda.empty_cache()



def contactmap_resnet101_rep(csv_name: str):
    """
    处理 CSV 文件并生成蛋白质嵌入表示。

    参数:
        csv_name (str): CSV 文件名（不带扩展名），例如 'new'。
    """
    # 定义默认参数
    args = {
        'input': Path(f'./data/datasets/{csv_name}.csv'),  # 输入 CSV 文件路径
        'id_col': 'Entry',                 # ID 列名
        'csv_sep': '\t',                   # CSV 文件分隔符
        'contactmap_dir': Path('./data/datasets_process/contactmap_rep/contactmap_8a'),  # 接触图目录
        'emb_model': 'resnet101',          # 使用的模型
        'emb_weight': 'IMAGENET1K_V2',    # 模型权重
    }

    # 自动检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print(f"使用设备: {device}")

    # 输出目录
    output_dir = Path('./data/datasets_process/contactmap_rep/contactmap_resnet101_rep')
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 读取 CSV 文件
    df = pd.read_csv(args['input'], sep=args['csv_sep'])
    
    # 定义接触图文件名格式
    contactmap_filename_format = '{}.npy'
    prot_ids = df[args['id_col']].unique().tolist()
    error_ids = []

    # 加载模型
    model = torchvision.models.get_model(args['emb_model'], weights=args['emb_weight'])
    model.fc = nn.Identity()
    model = model.eval().to(device)  # 将模型移动到设备

    # 处理每个蛋白质
    for prot_id in tqdm(prot_ids):
        contactmap_path = args['contactmap_dir'] / contactmap_filename_format.format(prot_id)
        if not contactmap_path.exists():
            error_ids.append(prot_id)
            continue

        # 加载接触图并生成嵌入
        contact_map = np.load(contactmap_path)
        cmap1 = np.array([contact_map, contact_map, contact_map])  # 转换为 3 通道

        with torch.no_grad():
            emb = model(torch.tensor(cmap1).unsqueeze(0).to(device, dtype=torch.float32)).cpu().squeeze(0)

        # 保存嵌入
        torch.save(emb, output_dir / f'{prot_id}.pt')

    if error_ids:  # 只有当 error_ids 不为空时才保存
        with open('./data/datasets_process/contactmap_rep/error_ids.txt', 'w') as f:
            f.write('\n'.join(error_ids))
        print(f"Processing is complete! Error IDs have been saved to 'error_ids.txt'")
    else:
        print("Processing complete! No error IDs found.")
        
        
def process_protein(prot_id):
    # esm2
    prot_esm2 = torch.load(f'./data/datasets_process/esm2_rep/{prot_id}.pt')['mean_representations'][36]
    # contactmap
    prot_contactmap = torch.load(f'./data/datasets_process/contactmap_rep/contactmap_resnet101_rep/{prot_id}.pt')
    # combine
    prot_combine = torch.cat((prot_esm2, prot_contactmap))
    
    emb = {'label': prot_id,
           'mean_representations': {36: prot_combine}
          }
    torch.save(emb, f'./data/esm_data/{prot_id}.pt')

def combine_esm2_contactmap_rep(csv_name):
    
    csv_path = Path(f'./data/datasets/{csv_name}.csv')
    data = pd.read_csv(csv_path, sep='\t') 
    prot_ids = data['Entry'].tolist()
    
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(process_protein, prot_ids), total=len(prot_ids)))
    
    print('combine complete..')


