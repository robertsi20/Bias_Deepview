import numpy as np

def tprs(sensitive_group, target, preds, n_classes):
    tpr_group_dict = {}
    unique_groups = np.unique(sensitive_group)

    for g in unique_groups:
        tpr_group_dict[g] = []

    for g in unique_groups:
        sensitive_g_filter = sensitive_group == g
        sub_preds = preds[sensitive_g_filter]
        gt = target[sensitive_g_filter]

        for cl in range(n_classes):
            # Use boolean indexing to filter predictions and ground truths
            preds_cl = sub_preds == cl
            gt_cl = gt == cl

            tp = np.sum(np.logical_and(preds_cl, gt_cl))
            group_rep = np.sum(gt_cl)

            if group_rep > 0:
                tpr = tp / group_rep
            else:
                tpr = 0

            tpr_group_dict[g].append(tpr)

    return tpr_group_dict


def calculate_gaps(tpr_dict, i, j):
    group_diff_dict = {}
    comparison = str(i) + "-" + str(j)
    group_diff_dict[comparison] = np.array(tpr_dict[i]) - np.array(tpr_dict[j])

    return group_diff_dict[comparison]