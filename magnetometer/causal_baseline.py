#!/usr/bin/env python3
"""Strictly causal harmonic baseline for magnetometer residual generation."""
from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import linalg

BASELINE_VERSION = "causal-baseline-v2"

def build_design_matrix(t_hours: np.ndarray, t_ref_min: float, t_ref_max: float) -> np.ndarray:
    t=np.asarray(t_hours,dtype=float); scale=max(float(t_ref_max)-float(t_ref_min),1e-12)
    tn=np.clip((t-float(t_ref_min))/scale,-0.5,1.5)
    return np.column_stack([np.ones_like(t),tn,tn**2,np.sin(2*np.pi*t/24),np.cos(2*np.pi*t/24),np.sin(2*np.pi*t/12),np.cos(2*np.pi*t/12),np.sin(2*np.pi*t/8),np.cos(2*np.pi*t/8),np.sin(2*np.pi*t/6),np.cos(2*np.pi*t/6)])

def robust_harmonic_fit(x: np.ndarray,cadence_s: float,t_hours: np.ndarray,n_iter: int=4,outlier_threshold_nt: float=30.0)->np.ndarray:
    values=np.asarray(x,dtype=float); t=np.asarray(t_hours,dtype=float); valid=np.isfinite(values)&np.isfinite(t)
    if valid.sum()<12:return np.full(11,np.nan)
    A=build_design_matrix(t[valid],float(np.min(t[valid])),float(np.max(t[valid]))); y=values[valid]; w=np.ones(y.size); c=np.zeros(A.shape[1])
    for _ in range(max(1,int(n_iter))):
        c,*_=linalg.lstsq(A*w[:,None],y*w); r=y-A@c; mad=float(np.median(np.abs(r-np.median(r)))); sigma=1.4826*mad+1e-12
        w=np.ones_like(r); w[np.abs(r)>float(outlier_threshold_nt)]=0.1; w[np.abs(r)>3*sigma]=0.01
    return c

def compute_causal_qdc_baseline(x: np.ndarray,cadence_s: float,*,fit_window_hours: float=24.0,update_minutes: float=15.0,min_history_fraction: float=0.50)->Tuple[np.ndarray,np.ndarray]:
    values=np.asarray(x,dtype=float)
    if values.ndim!=1: raise ValueError("x must be one-dimensional")
    if cadence_s<=0 or not np.isfinite(cadence_s): raise ValueError("cadence_s must be positive and finite")
    n=values.size; baseline=np.full(n,np.nan)
    if n==0:return baseline,values.copy()
    window=max(12,int(round(fit_window_hours*3600/cadence_s))); step=max(1,int(round(update_minutes*60/cadence_s))); t=np.arange(n)*cadence_s/3600
    last=None
    for block_start in range(window,n,step):
        hs=max(0,block_start-window); hist=values[hs:block_start]; th=t[hs:block_start]
        if hist.size==0 or float(np.isfinite(hist).mean())<min_history_fraction: continue
        coeff=robust_harmonic_fit(hist,cadence_s,th)
        if np.all(np.isfinite(coeff)): last=coeff
        if last is not None:
            block_end=min(n,block_start+step); A=build_design_matrix(t[block_start:block_end],float(th[0]),float(th[-1])); baseline[block_start:block_end]=A@last
    # Strictly causal warm-up fallback: sample t uses only t-1 and earlier.
    prev=pd.Series(values,dtype=float).shift(1)
    trailing=prev.rolling(max(1,min(window,n)),min_periods=1).median().to_numpy(dtype=float)
    missing=~np.isfinite(baseline); baseline[missing]=trailing[missing]
    residual=values-baseline; residual[~np.isfinite(values)]=np.nan
    return baseline,residual
