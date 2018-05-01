##-----------------------------------------------------------------------------
##
## challenge specific code and configuration
##
##-----------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
from sklearn.metrics import auc

def __nonlinear_interpolated_evalStats(block_df, blockWise_stats):
    """
    // given a block* of submitted belief score and blockwise statistics (see:__get_blockWise_stats)
    calculates the Precision, Recall & False Positive Rate
    *A block by definition should have the same belief score

    """
    blockValue = block_df.predict.unique()
    if len(blockValue) != 1:
        raise Exception("grouping by predict column doesnt yield unique predict vals per group..WIERD")
    blockValue = blockValue[0]
    blockStats = blockWise_stats[blockWise_stats.blockValue == blockValue].squeeze() #squeeze will convert one row df to series

    block_precision = []
    block_recall = []
    block_fpr = []

    block_true_pos = []
    block_pred_pos = []
    block_true_neg = []
    block_actual_pos = []

    total_elements = blockWise_stats.cum_numElements.max()
    total_truePos = blockWise_stats.cum_truePos.max()
    total_trueNeg = total_elements - total_truePos
    for block_depth, row in enumerate(block_df.iterrows()):
        block_depth += 1  #increase block depth by 1
        row = row[1]
        #calculate the cumulative true positives seen till the last block from the current active block
        # and the total number of elements(cumulative) seen till the last block
        if blockStats.block == 1: #no previous obviously
            cum_truePos_till_lastBlock = 0
            cum_numElements_till_lastBlock = 0
            cum_trueNeg_till_lastBlock = 0
        elif blockStats.block > 1:
            last_blockStats = blockWise_stats[blockWise_stats.block == (blockStats.block-1)].squeeze()
            cum_truePos_till_lastBlock = last_blockStats['cum_truePos']
            cum_numElements_till_lastBlock = last_blockStats['cum_numElements']
            cum_trueNeg_till_lastBlock = cum_numElements_till_lastBlock - cum_truePos_till_lastBlock

        truePos = cum_truePos_till_lastBlock + (blockStats.block_truePos_density*block_depth)
        falsePos = cum_trueNeg_till_lastBlock + ((1 - blockStats.block_truePos_density ) * block_depth)

        #precision
        interpolated_precision = truePos /(cum_numElements_till_lastBlock+block_depth)
        block_precision.append(interpolated_precision)
        #recall == true positive rate
        interpolated_recall = truePos /total_truePos
        block_recall.append(interpolated_recall)
        #fpr == false positive rate
        interpolated_fpr = falsePos / total_trueNeg
        block_fpr.append(interpolated_fpr)

    block_df['precision'] = block_precision
    block_df['recall'] = block_recall
    block_df['fpr'] = block_fpr
    block_df['block_depth'] = np.arange(1,block_df.shape[0]+1)
    block_df['block'] = blockStats.block
    return(block_df)


def __get_blockWise_stats(sub_stats):
    """
    calculate stats for each block of belief scores
    """
    pd.options.mode.chained_assignment = None
    feature_col_range = int(sub_stats.shape[1] / 2)
    #group to calculate group wise stats for each block
    results = []
    for i in range(feature_col_range):
        sub_stats_subset = sub_stats[[i,i+feature_col_range]]
        sub_stats_subset.columns = ['predict', 'truth']
        sub_stats_subset = sub_stats_subset.sort_values('predict', ascending=False)
        grouped = sub_stats_subset.groupby(['predict'], sort=False)

        #instantiate a pandas dataframe to store the results for each group (tied values)
        result = pd.DataFrame.from_dict({'block':range(len(grouped)),
                                             'block_numElements'  : np.nan,
                                             'block_truePos_density' : np.nan,
                                             'block_truePos'      : np.nan,
                                             'blockValue'   : np.nan
                                             })

        for block,grp in enumerate(grouped):
            name,grp = grp[0],grp[1]
            truePositive = sum(grp.truth == 1)
            grp_truePositive_density = truePositive / float(len(grp))
            idxs = result.block == block
            result.block_truePos_density[idxs] = grp_truePositive_density
            result.block_numElements[idxs] = len(grp)
            result.block_truePos[idxs] = truePositive
            result.blockValue[idxs] = grp.predict.unique()
        result.block = result.block + 1
        result['cum_numElements'] = result.block_numElements.cumsum()
        result['cum_truePos'] = result.block_truePos.cumsum()
        results.append(result)
    return results

def getAUROC_PR(sub_stats):
    #calculate blockwise stats for tied precdiction scores
    blockWise_stats = __get_blockWise_stats(sub_stats)
    feature_col_range = sub_stats.shape[1] / 2

    results = []
    for i in range(len(blockWise_stats)):
        sub_stats_subset = sub_stats[[i,i+feature_col_range]]
        sub_stats_subset.columns = ['predict', 'truth']
        sub_stats_subset = sub_stats_subset.sort_values('predict', ascending=False)
        #calculate precision recall & fpr for each block
        grouped = sub_stats_subset.groupby(['predict'], sort=False)
        ss = grouped.apply(__nonlinear_interpolated_evalStats,blockWise_stats[i])

        precision, recall,  fpr, threshold = ss.precision, ss.recall, ss.fpr, ss.predict
        tpr = recall #(Recall and True positive rates are same)
        roc_auc = auc(fpr,tpr,reorder=True)
        #PR curve AUC (Fixes error when prediction == truth)
        recall_new=list(recall)
        precision_new=list(precision)
        recall_new.reverse()
        recall_new.append(0)
        recall_new.reverse()

        precision_new.reverse()
        precision_new.append(precision_new[len(precision_new)-1])
        precision_new.reverse()

        PR_auc = auc(recall_new, precision_new,reorder=True)
        #results = [ round(x,4) for x in results]
        results.append(PR_auc)
    return results
