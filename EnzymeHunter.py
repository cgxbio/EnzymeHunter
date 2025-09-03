import os
import sys
import pandas as pd
import torch
import numpy as np
import csv
import re
import argparse
print(os.getcwd())
sys.path.append('./src')


from model import EnzNonEnzBinary,ProteinDataset
from torch.utils.data import DataLoader
from utils import (fasta_to_csv, retrive_esm2_embedding_36_mean, 
                  process_fasta_and_save_contact_maps, contactmap_resnet101_rep,
                  combine_esm2_contactmap_rep, csv_to_fasta_test_dataset)
from infer import use_maxsep_pred_ec_number

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def prepare_data(test_name):
    """Prepare data and generate features"""
    fasta_path = f'./data/datasets/{test_name}.fasta'
    csv_path = f'./data/datasets/{test_name}.csv'
    fasta_to_csv(fasta_path, csv_path)
    retrive_esm2_embedding_36_mean(test_name)
    process_fasta_and_save_contact_maps(test_name)
    contactmap_resnet101_rep(test_name)
    combine_esm2_contactmap_rep(test_name)

def predict_enz_nonenz(test_name, model_path='./model/train_model/enz_nonenz_pred_model.pt'):
    """Predict enzyme/non-enzyme classification"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_csv_path = f'./data/datasets/{test_name}.csv'
    test_df = pd.read_csv(test_csv_path, sep='\t')
    test_dataset = ProteinDataset(test_df, './data/esm_data')
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, 
                           num_workers=8, pin_memory=True)

    model = EnzNonEnzBinary(256, 64, 0.1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device, non_blocking=True)
            probs = model(features)
            all_probs.append(probs.cpu())
    
    test_df['pred_prob'] = torch.cat(all_probs).numpy()
    test_df['pred_label'] = (test_df['pred_prob'] > 0.5).astype(int)
    
    os.makedirs('./results/tmp', exist_ok=True)
    output_path = f'./results/tmp/{test_name}_pred_enz_nonenz.csv'
    test_df.to_csv(output_path, sep='\t', index=False)
    print(f"✅ Enzyme/Non-enzyme predictions saved: {output_path}")
    
    torch.cuda.empty_cache()
    del model  
    
    return test_df

def prepare_enz_subset(test_name, pred_results):
    """Filter and prepare enzyme subset for EC number prediction"""
    enz_results = pred_results[pred_results['pred_label'] == 1].iloc[:, :2].reset_index(drop=True)

    datasets_path = './data/datasets'
    pred_enz_path = os.path.join(datasets_path, f'{test_name}_pred_enz.csv')
    pred_enz_fasta_path = os.path.join(datasets_path, f'{test_name}_pred_enz.fasta')
    
    enz_results.to_csv(pred_enz_path, sep='\t', index=False)
    csv_to_fasta_test_dataset(pred_enz_path, pred_enz_fasta_path)
    
    return enz_results

def combine_maxsep_diamond_tmvec(
    test_fasta_path,
    test_csv_path,
    prediction_csv_path,
    diamond_db_path,
    diamond_exec_path,
    diamond_output_path,
    tmvec_output_path,
    tmvec_output_embedding_path
):  
    
    diamond_command = (
        f"{diamond_exec_path}diamond blastp -d {diamond_db_path} "
        f"-q {test_fasta_path} -o {diamond_output_path} --quiet -p 32 --header --ultra-sensitive"
    )
    os.system(diamond_command)
    
    tmvec_command = (
        f"tmvec-search "
        f"--query {test_fasta_path} "
        f"--tm-vec-model ./model/diamond_tmvec_database/tm_vec_swiss_model_large.ckpt "
        f"--tm-vec-config ./model/diamond_tmvec_database/tm_vec_swiss_model_large_params.json "
        f"--database ./model/diamond_tmvec_database/bagel_database/db.npy "
        f"--metadata ./model/diamond_tmvec_database/bagel_database/meta.npy "
        f"--database-fasta ./data/datasets/split100.fasta "
        f"--device 0 "
        f"--output-format tabular "
        f"--output {tmvec_output_path} "
        f"--output-embeddings {tmvec_output_embedding_path}  "
        f"--protrans-model ./model/diamond_tmvec_database/prot_t5_xl_uniref50 "
    )
    os.system(tmvec_command)

    test_true = pd.read_csv(test_csv_path, sep="\t")
    test_true = test_true.iloc[:, [0, 1]].rename(columns={"EC number": "EC number_true"})
    entry_order = []
    with open(test_csv_path, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            entry_order.append(row['Entry'])

    predictions_dict = {}
    with open(prediction_csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            if len(row) < 2:
                continue
            seqid = row[0]
            ec_string = ','.join(row[1:])
            ec_matches = re.findall(r'EC:(\d+\.\d+\.\d+\.(?:n\d+|\d+))', ec_string)
            if ec_matches:
                if seqid not in predictions_dict:
                    predictions_dict[seqid] = []
                predictions_dict[seqid].extend(ec_matches)

    data = []
    for entry in entry_order:
        ec_numbers = ""
        if entry in predictions_dict:
            ec_numbers = ";".join(predictions_dict[entry])
        data.append({'Entry': entry, 'EC number_pred': ec_numbers})
    test_pred = pd.DataFrame(data)

    test_merge = test_true.merge(test_pred, on="Entry")

    with open(diamond_output_path, 'r') as f:
        blast_file = f.readlines()

    blast_header = blast_file[2].strip().split(':')[1].split(',')
    blast_header = [i.strip() for i in blast_header]
    blast_results = pd.read_csv(diamond_output_path, skiprows=3, sep='\t')
    blast_results.columns = blast_header

    split100 = pd.read_csv("./data/datasets/split100.csv", sep="\t")
    diamond_pred = blast_results.merge(split100, left_on='Subject ID', right_on='Entry')
    diamond_pred = diamond_pred.iloc[:, [0, 1, 2, 13]]

    diamond_pred_top1 = diamond_pred.sort_values('Percentage of identical matches', ascending=False).groupby('Query ID').head(1)
    diamond_pred_top1 = diamond_pred_top1.iloc[:, [0, 2, 3]].rename(columns={
        'EC number': 'EC number_diamond',
        "Percentage of identical matches": "identity",
        'Query ID': "Entry"
    })

    test_merge = test_merge.merge(diamond_pred_top1, on='Entry', how='left')

    test_tmvec = pd.read_table(tmvec_output_path)
    test_tmvec = test_tmvec.merge(split100, left_on="database_id", right_on='Entry').iloc[:, [0, 2, 3, 5]]

    test_tmvec_top1 = test_tmvec.drop_duplicates(subset='query_id', keep='first').rename(columns={
        'EC number': 'EC number_tmvec',
        'query_id': "Entry"
    })
    test_tmvec_top1 = test_tmvec_top1.iloc[:, [0, 2, 3]]

    test_merge = test_merge.merge(test_tmvec_top1, on='Entry')

    test_merge['EC number'] = test_merge.apply(
        lambda row: row['EC number_tmvec'] if (row['identity'] > 90 and row['tm-score'] > 0.9)
        else row['EC number_pred'], axis=1
    )

    final_test_results = test_merge.iloc[:, [0, 7]]
    
    return final_test_results

def predict_ec_numbers(test_name):
    """Run EC number prediction pipeline"""
    test_enz_name = f'{test_name}_pred_enz'
    use_maxsep_pred_ec_number("split100", test_enz_name, 
                             model_name="ec_number_pred_model")
    
    paths = {
        'test_fasta_path': f'./data/datasets/{test_enz_name}.fasta',
        'test_csv_path': f'./data/datasets/{test_enz_name}.csv',
        'prediction_csv_path': f'./results/tmp/{test_enz_name}_maxsep.csv',
        'diamond_db_path': './model/diamond_tmvec_database/split100.dmnd',
        'diamond_exec_path': './model/diamond_tmvec_database/',
        'diamond_output_path': f'./results/diamond_tmvec_output/{test_enz_name}.tsv',
        'tmvec_output_path': f'./results/diamond_tmvec_output/{test_enz_name}.txt',
        'tmvec_output_embedding_path': f'./results/diamond_tmvec_output/{test_enz_name}_embedding.npy'
    }
    
    results = combine_maxsep_diamond_tmvec(**paths)
    results.to_csv(f'./results/tmp/{test_enz_name}.csv', sep='\t', index=False)
    
    return results

def merge_final_results(test_name, all_enzymes=False):
    """Combine enzyme/non-enzyme and EC number predictions"""
    if all_enzymes:
        enz_nonenz = pd.read_csv(f'./data/datasets/{test_name}.csv', sep='\t')
        enz_nonenz['pred_prob'] = 1.0
        enz_nonenz['pred_label'] = 1
    else:
        enz_nonenz = pd.read_csv(f'./results/tmp/{test_name}_pred_enz_nonenz.csv', sep='\t')
    
    enz_ec = pd.read_csv(f'./results/tmp/{test_name}_pred_enz.csv', sep='\t')

    merged = pd.merge(enz_nonenz, enz_ec, on='Entry', how='left')
    merged['pred_EC number'] = np.where(
        merged['pred_label'] == 1,
        merged['EC number'],
        '0.0.0.0'
    )
    
    final = merged[['Entry', 'Sequence', 'pred_prob', 'pred_label', 'pred_EC number']]
    final.to_csv(f'./results/{test_name}_final_pred_results.csv', sep='\t', index=False)
    
    return final

def run_prediction_pipeline(test_name, all_enzymes=False):
    """Main function to run the complete prediction pipeline"""
    print(f"Starting prediction pipeline for {test_name}")
    print(f"All enzymes mode: {'ON' if all_enzymes else 'OFF'}")
    print("Preparing data and generating features...")
    prepare_data(test_name)
    
    if not all_enzymes:
        print("Predicting enzyme/non-enzyme classification...")
        enz_nonenz_results = predict_enz_nonenz(test_name)
        print("Filtering enzyme subset...")
        enz_subset = prepare_enz_subset(test_name, enz_nonenz_results)
    else:
        print("Skipping enzyme/non-enzyme prediction (all-enzymes mode)")
        original_csv = f'./data/datasets/{test_name}.csv'
        enz_subset = pd.read_csv(original_csv, sep='\t')
        enz_subset['pred_prob'] = 1.0
        enz_subset['pred_label'] = 1
        os.makedirs('./results/tmp', exist_ok=True)
        enz_nonenz_path = f'./results/tmp/{test_name}_pred_enz_nonenz.csv'
        enz_subset.to_csv(enz_nonenz_path, sep='\t', index=False)
    
    print("Predicting EC numbers...")
    ec_results = predict_ec_numbers(test_name)
    
    print("Merging final results...")
    final_results = merge_final_results(test_name, all_enzymes)
    
    print(f"✅ Pipeline completed! Results saved to ./results/{test_name}_final_pred_results.csv")
    return final_results

def main():
    parser = argparse.ArgumentParser(description="Run EnzymeHunter prediction pipeline.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the test dataset (e.g., 'test_dataset_enz_nonenz')"
    )
    parser.add_argument(
        "--all_are_enzymes",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=False,
        help="Set to True if all proteins are known to be enzymes (default: False)"
    )
    args = parser.parse_args()
    final_results = run_prediction_pipeline(args.dataset, args.all_are_enzymes)
    print("\nFinal Results:")
    print(final_results.head())
    
if __name__ == "__main__":
    main()