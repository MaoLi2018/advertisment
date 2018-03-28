# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:06:31 2018

@author: maoli
"""

import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from os.path import join
from scipy import special
from functools import reduce

def margins(a):
    margsums = []
    ranged = list(range(a.ndim))
    for k in ranged:
        marg = np.apply_over_axes(np.sum, a, [j for j in ranged if j != k])
        margsums.append(marg)
    return margsums

def expected_freq(observed):
    observed = np.asarray(observed, dtype=np.float64)
    margsums = margins(observed)
    d = observed.ndim
    expected = reduce(np.multiply, margsums) / observed.sum() ** (d - 1)
    return expected

def chi2_contingency(observed, correction=True):
    observed = np.asarray(observed) + 0.0001
    if np.any(observed < 0):
        raise ValueError("All values in `observed` must be nonnegative.")
    if observed.size == 0:
        raise ValueError("No data; `observed` has size 0.")
    expected = expected_freq(observed)
    dof = expected.size - sum(expected.shape) + expected.ndim - 1
    if dof == 0:
        chi2 = 0.0
        p = 1.0
    else:
        chi2 = ((observed - expected) ** 2 / expected).sum()
        p = special.chdtrc(dof, chi2)    
    return chi2, p, dof, expected



def woe_calc(bad, good, goodfreq, badfreq):
    target_rt = float(bad) / float(badfreq)
    non_target_rt = float(good) / float(goodfreq)
    if float(bad) != 0.0 and float(bad) / (float(bad) + float(good)) != 1.0:
        woe = np.log(float(target_rt / non_target_rt))
    ### use 2 to fix the outlier
    elif target_rt == 0.0:
        woe = -2.0
    elif float(bad) / (float(bad) + float(good)) == 1.0:
        woe = 2.0
    return woe




def calc_nominal_woe(df, var, tgt, bins=3,small=0.02):
    if df[var].count() == 0:
        return None, None, None
    vval_na = 'NA'
    i = 0
    while vval_na in df[var].values:
        vval_na = 'NA_{:02d}'.format(i)
        i += 1
    df.fillna({var: vval_na}, inplace=True)
    col_t = [c for c in df.columns if c != var and c != tgt][0]
    ds = df[[var, tgt, col_t]].groupby([var, tgt]).count().unstack().fillna(value=0)
    ds.columns = ds.columns.droplevel(0)
    ds.reset_index(inplace=True)
    ds[var] = ds[var].apply(lambda v: [v])
    is_small = ((ds[0] + ds[1]) / df.shape[0]) < small
    while sum(is_small) > 0:
        small_sum = ds.loc[is_small].sum()
        ds.drop(ds[is_small].index, inplace=True)
        ds = ds.append(pd.DataFrame(small_sum).transpose(), ignore_index=True)
        pop_rto = ((ds[0] + ds[1]) / df.shape[0])
        is_small = pop_rto < small
        if sum(is_small) == 1:
            pop_rto = pop_rto.astype(float)
            is_small[pop_rto[:-1].argmin()] = True
    ds['dft_rto'] = ds[1] / ds[[0, 1]].sum(axis=1)
    ds = ds.sort_values('dft_rto').reset_index(drop=True)
    try:
        df.replace({var: {vval_na: np.nan}}, inplace=True)
    except TypeError:
        pass
    chisq = []
    for i in range(ds.shape[0] - 1):
        chisq.append(chi2_contingency(ds.iloc[[i, i + 1]][[0, 1]])[0])
    chisq.append(9999999.0)
    ds['chisq'] = chisq
    '''
    #------- chimerge: merge the adjacent bins -------#
    '''
    while (ds.shape[0] >= bins*3) or (ds.shape[0] > 2 and ds.chisq.min() <= stats.chi2.ppf(0.95, 1)):
        ds_idx_list = list(ds.index)
        k = ds_idx_list.index(ds[ds.chisq == ds.chisq.min()].index[0])
        ds.ix[ds_idx_list[k], [var, 0, 1]] = ds.ix[[ds_idx_list[k], ds_idx_list[k + 1]], [var, 0, 1]].sum()
        ds.ix[ds_idx_list[k], 'dft_rto'] = 1.0 * ds.ix[ds_idx_list[k], 1] / (
            ds.ix[ds_idx_list[k], 0] + ds.ix[ds_idx_list[k], 1])
        ds = ds.drop(ds_idx_list[k + 1], axis=0)
        ds = ds.reset_index(drop=True)
        
        if k != 0:
            ds.ix[ds_idx_list[k - 1], 'chisq'] = chi2_contingency(ds.iloc[ds_idx_list[k - 1:k + 1]][[0, 1]])[0]
        if k < ds.shape[0] - 1:
            ds.ix[ds_idx_list[k], 'chisq'] = chi2_contingency(ds.iloc[ds_idx_list[k:k + 2]][[0, 1]])[0]
        else:
            ds.ix[ds_idx_list[k], 'chisq'] = 9999999.0
        
    
    '''
    #-------- chimerge: control bin size -------#
    '''
    pop_cut = float(ds[0].sum() + ds[1].sum()) / 50
    ds['pop'] = ds[0] + ds[1]
    # print pop_cut
    while (ds.shape[0] >= bins 
           and (ds['pop'].min() < pop_cut
                or ds.chisq.min() <= stats.chi2.ppf(0.99, 1))) or ds[1].min() <= 0:
        # calculate chisquare statistic
        chisq = []
        for i in range(ds.shape[0] - 1):
            chisq.append(chi2_contingency(ds.iloc[[i, i + 1]][[0, 1]])[0])
        chisq.append(9999999.0)
        ds['chisq'] = chisq
        # locate the smallest size by index
        ds_idx_list = list(ds.index)
        k = ds_idx_list.index(ds[ds['pop'] == ds['pop'].min()].index[0])
        if k == len(ds_idx_list) - 1:
            k -= 1
        elif ds.ix[ds_idx_list[k], 'chisq'] > ds.ix[ds_idx_list[k - 1], 'chisq']:
            k -= 1
        # merge the adjacent bins, drop the second bin
        ds.ix[ds_idx_list[k], [var, 0, 1, 'pop']] = ds.ix[
            [ds_idx_list[k], ds_idx_list[k + 1]], [var, 0, 1, 'pop']].sum()
        ds.ix[ds_idx_list[k], 'dft_rto'] = 1.0 * ds.ix[ds_idx_list[k], 1] / (
            ds.ix[ds_idx_list[k], 0] + ds.ix[ds_idx_list[k], 1])
        ds = ds.drop(ds_idx_list[k + 1], axis=0)
        ds = ds.reset_index(drop=True)
    '''
    #------- Calculate woe & IV -------#
    '''
    goodfreq = ds[0].sum()
    badfreq = ds[1].sum()
    ds[var + '_cwoe'] = ds.apply(lambda x: woe_calc(x[1], x[0], goodfreq, badfreq), axis=1)
    # IVs
    ds[var + '_IVs'] = ds.apply(lambda x: x[var + '_cwoe'] * (float(x[1]) / badfreq - float(x[0]) / goodfreq), axis=1)
    IV = ds[var + '_IVs']
    # Set stats of other's bin the same as that of the bin where least value locates
    # For data has not appeared
    vval_min = df[var].value_counts().argmin()
    ds = pd.concat([ds, ds.loc[ds[var].apply(lambda v_lst: vval_min in v_lst)]], ignore_index=True)
    ds.ix[ds.tail(1).index, var] = 'others'
    # Set var column (1st) to the last (to be readable)
    ds = ds.reindex_axis(list(ds.columns[1:]) + [ds.columns[0]], axis=1)
    '''
    #------- get the reference table -------#
    '''
    def is_in_bin(v_lst, v_val, vval_na):
        if 'others' == v_lst:
            return False
        if pd.isnull(v_val):
            return vval_na in v_lst
        else:
            return v_val in v_lst
    uniq_val = df[var].unique()
    ref_dict = {}
    for v_val in uniq_val:
        ref_dict[v_val] = ds.ix[ds[var].apply(lambda v_lst: is_in_bin(v_lst, v_val, vval_na)), var + '_cwoe'].values[0]
    ref_dict['others'] = 0  # ds.ix[ds[var] == 'others', var + "_cwoe"].values[0]
    ref_table = pd.DataFrame.from_dict(ref_dict, orient='index')
    ref_table = ref_table.reset_index().rename(columns={'index': 'var_value', 0: 'woe'})
    ref_table['var_name'] = var
    ref_table = ref_table.reindex_axis(['var_name', 'var_value', 'woe'], axis=1)
    ds.loc[ds[var]=='others',[0,1,'dft_rto','chisq','pop',var+'_cwoe',var+'_IVs']]=0
    IV = sum(ds[var + '_IVs'])
    return ref_table, ds[var + '_IVs'], ds, IV



























