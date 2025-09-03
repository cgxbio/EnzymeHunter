import torch
import torch.nn as nn
import torch.nn.functional as F
 
from utils import hierarchical_similarity , compute_similarity, dynamic_margin_multilabel
 
def SupConHardLoss(model_emb, temp, n_pos):

    features = F.normalize(model_emb, dim=-1, p=2)
    features_T = torch.transpose(features, 1, 2)
    anchor = features[:, 0]
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 
    anchor_dot_features = anchor_dot_features.squeeze(1)
    logits = anchor_dot_features - 1/temp

    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) 
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) 
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos
    loss = - log_prob.mean()
    return loss    

class HierarchicalTripletLoss(nn.Module):
    def __init__(self, base_margin=1.0, similarity_mode="max"):
        super(HierarchicalTripletLoss, self).__init__()
        self.base_margin = base_margin
        self.similarity_mode = similarity_mode
        
        self.triplet_loss = nn.TripletMarginLoss(margin=1, reduction='mean')

    def forward(self, anchor, positive, negative, ec_anchor_pos_neg):

        ec_anchor = [item['anchor_ec'] for item in ec_anchor_pos_neg]
        ec_positive = [item['pos_ec'] for item in ec_anchor_pos_neg]
        ec_negative = [item['neg_ec'] for item in ec_anchor_pos_neg]
            
        pos_distance = F.pairwise_distance(anchor, positive)
        neg_distance = F.pairwise_distance(anchor, negative)

        sim_pos = torch.tensor([compute_similarity(ec_a, ec_p, self.similarity_mode) for ec_a, ec_p in zip(ec_anchor, ec_positive)], device=anchor.device)
        sim_neg = torch.tensor([compute_similarity(ec_a, ec_n, self.similarity_mode) for ec_a, ec_n in zip(ec_anchor, ec_negative)], device=anchor.device)
        
        margins = dynamic_margin_multilabel(sim_pos, sim_neg, self.base_margin)

        losses =  torch.max(pos_distance - neg_distance + margins, torch.tensor(0.0))

        return losses.mean()



class combined_loss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5, similarity_mode="max"):
        super(combined_loss, self).__init__()

        self.triplet_loss = HierarchicalTripletLoss(base_margin=margin, similarity_mode=similarity_mode)
        self.bce_loss = nn.BCELoss()
        self.alpha = alpha

    def forward(self, anchor_embedding, positive_embedding, negative_embedding, 
                anchor_pred_ec, anchor_one_hots_tensor,
                pos_pred_ec,pos_one_hots_tensor,
                ang_pred_ec,neg_one_hots_tensor,
                ec_anchor_pos_neg):

        triplet_loss = self.triplet_loss(anchor_embedding, positive_embedding, negative_embedding, 
                                         ec_anchor_pos_neg)
        
        bce_loss = self.bce_loss(anchor_pred_ec, anchor_one_hots_tensor)+self.bce_loss(pos_pred_ec,pos_one_hots_tensor)+self.bce_loss(ang_pred_ec,neg_one_hots_tensor)
        total_loss = self.alpha * triplet_loss + (1 - self.alpha) * bce_loss
        
        return total_loss   

    
    
