import torch
from torch.optim import LBFGS
import numpy as np
import pandas as pd

def plackett_luce_loss(models, X_unseen):
    all_predictions3 = np.zeros((len(X_unseen), len(models)))
    all_predictions2 = np.zeros((len(X_unseen), len(models)))
    all_predictions1 = np.zeros((len(X_unseen), len(models)))
    all_predictions0 = np.zeros((len(X_unseen), len(models)))

    for i, model in enumerate(models):
        y_pred_proba3 = model.predict_proba(X_unseen)[:, 3]
        all_predictions3[:, i] = y_pred_proba3

        y_pred_proba2 = model.predict_proba(X_unseen)[:, 2]
        all_predictions2[:, i] = y_pred_proba2

        y_pred_proba1 = model.predict_proba(X_unseen)[:, 1]
        all_predictions1[:, i] = y_pred_proba1

        y_pred_proba0 = model.predict_proba(X_unseen)[:, 0]
        all_predictions0[:, i] = y_pred_proba0

    final_predictions0 = np.mean(all_predictions0, axis=1)
    final_predictions1 = np.mean(all_predictions1, axis=1)
    final_predictions2 = np.mean(all_predictions2, axis=1)
    final_predictions3 = np.mean(all_predictions3, axis=1)

    return [final_predictions0, final_predictions1, final_predictions2, final_predictions3]

def plan_scores_def(cured_data, plackett_luce, test):
    losses = plackett_luce_loss(plackett_luce, test)
    loss_df = pd.DataFrame(losses).T.idxmax(axis=1)
    preds_data = pd.DataFrame(cured_data)
    preds_data['pl'] = loss_df
    preds_data.columns = ['A', 'B', 'Wins', 'Tier']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unique_plans = np.unique(np.concatenate((preds_data['A'], preds_data['B'])))
    plan_to_index = {plan: index for index, plan in enumerate(unique_plans)}

    preds_data['A_idx'] = preds_data['A'].map(plan_to_index)
    preds_data['B_idx'] = preds_data['B'].map(plan_to_index)

    initial_scores = np.ones(len(unique_plans))
    scores = torch.tensor(initial_scores, dtype=torch.float64, device=device, requires_grad=True)

    plan_tiers = {}
    for plan in unique_plans:
        tier_entry = preds_data[preds_data['A'] == plan]['Tier'].values
        if len(tier_entry) > 0:
            plan_tiers[plan] = tier_entry[0]
        else:
            plan_tiers[plan] = 0

    tiers = torch.tensor([plan_tiers.get(plan, 0) for plan in unique_plans], dtype=torch.float64, device=device)

    def bradley_terry_likelihood():
        log_likelihood = 0
        for i, row in preds_data.iterrows():
            a = int(row['A_idx'])
            b = int(row['B_idx'])
            p = scores[a] / (scores[a] + scores[b])
            log_likelihood += row['Wins'] * torch.log(p) + (1 - row['Wins']) * torch.log(1 - p)
        return -log_likelihood

    def apply_tier_adjustments(scores, tiers):
            adjusted_scores = scores.clone()
            for i in range(len(scores)):
                if tiers[i] == 3:
                    adjusted_scores[i] *= 64
                    adjusted_scores[i] += 60#50
                elif tiers[i] == 2:
                    adjusted_scores[i] *= 16
                    adjusted_scores[i] += 30#20
                elif tiers[i] == 1:
                    adjusted_scores[i] *= 4
                    adjusted_scores[i] += 8#5
                elif tiers[i] == 0:
                    adjusted_scores[i] *= 1
                    adjusted_scores[i] += 1
            return adjusted_scores

    optimizer = LBFGS([scores], lr=1)

    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_likelihood()
        loss.backward()
        return loss

    optimizer.step(closure)

    optimized_scores = scores.cpu().detach().numpy()
    adjusted_scores = apply_tier_adjustments(scores, tiers).cpu().detach().numpy()

    plan_scores = {plan: adjusted_scores[plan_to_index[plan]] for plan in unique_plans}

    return plan_scores