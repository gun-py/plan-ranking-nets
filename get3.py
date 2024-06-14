import torch
from torch.optim import LBFGS
import numpy as np
import pandas as pd

def plan_scores_def(cured_data, plackett_luce, test):
    def plackett_luce_loss_optim(models, X_unseen):
        all_predictions = np.zeros((len(X_unseen), len(models), 4))

        for i, model in enumerate(models):
            y_pred_proba = model.predict_proba(X_unseen)
            all_predictions[:, i, :] = y_pred_proba

        final_predictions = np.mean(all_predictions, axis=1)
        return [final_predictions[:, 0], final_predictions[:, 1], final_predictions[:, 2], final_predictions[:, 3]]

    losses = plackett_luce_loss_optim(plackett_luce, test)
    loss_df = pd.DataFrame(losses).T.idxmax(axis=1)
    preds_data = pd.DataFrame(cured_data)
    preds_data['pl'] = loss_df
    preds_data.columns = ['A', 'B', 'Wins', 'Segs']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unique_plans = np.unique(np.concatenate((preds_data['A'], preds_data['B'])))
    plan_to_index = {plan: index for index, plan in enumerate(unique_plans)}

    preds_data['A_idx'] = preds_data['A'].map(plan_to_index)
    preds_data['B_idx'] = preds_data['B'].map(plan_to_index)

    initial_scores = np.ones(len(unique_plans))
    scores = torch.tensor(initial_scores, dtype=torch.float64, device=device, requires_grad=True)

    plan_segs = {}
    for plan in unique_plans:
        segs_entry = preds_data[preds_data['A'] == plan]['Segs'].values
        if len(segs_entry) > 0:
            plan_segs[plan] = segs_entry[0]
        else:
            plan_segs[plan] = 0

    segs = torch.tensor([plan_segs.get(plan, 0) for plan in unique_plans], dtype=torch.float64, device=device)

    def bradley_terry_likelihood():
        log_likelihood = 0
        for i, row in preds_data.iterrows():
            a = int(row['A_idx'])
            b = int(row['B_idx'])
            p = scores[a] / (scores[a] + scores[b])
            log_likelihood += row['Wins'] * torch.log(p) + (1 - row['Wins']) * torch.log(1 - p)
        return -log_likelihood

    def apply_gradclip_adj(scores, segs):
            gradclip_scores = scores.clone()
            for i in range(len(scores)):
                if segs[i] == 3:
                    gradclip_scores[i] *= 64
                    gradclip_scores[i] += 60#50
                elif segs[i] == 2:
                    gradclip_scores[i] *= 16
                    gradclip_scores[i] += 30#20
                elif segs[i] == 1:
                    gradclip_scores[i] *= 4
                    gradclip_scores[i] += 8#5
                elif segs[i] == 0:
                    gradclip_scores[i] *= 1
                    gradclip_scores[i] += 1
            return gradclip_scores

    optimizer = LBFGS([scores], lr=1)

    def closure():
        optimizer.zero_grad()
        loss = bradley_terry_likelihood()
        loss.backward()
        return loss

    optimizer.step(closure)

    optimized_scores = scores.cpu().detach().numpy()
    raw_scores = apply_gradclip_adj(scores, segs).cpu().detach().numpy()

    plan_scores = {plan: raw_scores[plan_to_index[plan]] for plan in unique_plans}

    return plan_scores