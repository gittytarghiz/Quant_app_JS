# portfolio_optimization/walkforward_nsga2.py
import numpy as np, pandas as pd
from pathlib import Path; import sys
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent; sys.path.append(str(ROOT))
from data_management.monolith_loader import get_downloaded_series

def walkforward_nsga2(
    tickers, start, end, dtype="close", interval="1d", rebalance="monthly",
    min_weight=0.0, max_weight=1.0, min_obs=60, leverage=1.0,
    objectives=None, pop=64, gens=12, eta_c=12., eta_m=20., mut_rate=0.1,
    alpha=0.2, seed=42, **_
) -> dict:
    """Vectorized NSGA-II with Ledoit–Wolf shrinkage covariance. Executes max-Sharpe from Pareto front."""
    rng=np.random.default_rng(seed)
    prices=get_downloaded_series(tickers,start,end,dtype=dtype,interval=interval).dropna()
    rets=prices.pct_change().dropna()
    if rets.empty: raise ValueError("No returns available.")
    freq={"daily":"D","weekly":"W-FRI","monthly":"ME","quarterly":"Q"}[rebalance]
    rbd=[d for d in pd.date_range(rets.index[0],rets.index[-1],freq=freq) if d in rets.index] or [rets.index[0]]
    n=len(tickers); lo,hi=float(min_weight),float(max_weight)
    pnl=pd.Series(0.,index=rets.index); W=[]
    obj=objectives or ["ann_vol","cvar"]

    for t in rbd:
        win=rets.loc[:t]
        if len(win)<min_obs: w_best=np.full(n,1/n)
        else:
            R=win.values; S=LedoitWolf().fit(R).covariance_   # shrinkage cov
            # init
            X=rng.uniform(lo,hi,(pop,n)); X/=X.sum(1,keepdims=1); V=rng.normal(0,0.05,(pop,n))
            def eval_metrics(Wm):
                P=R@Wm.T; m=P.mean(0); s=P.std(0)+1e-12
                ann_ret=252*m; ann_vol=np.sqrt(252)*s
                sharpe=(m/s)*np.sqrt(252); cvar=np.partition(-P,int(alpha*P.shape[0]),axis=0)[-int(alpha*P.shape[0]):].mean(0)
                return {"ann_return":ann_ret,"ann_vol":ann_vol,"sharpe":sharpe,"cvar":cvar}
            def scores(M): return np.vstack([M[k] if k in {"ann_vol","cvar"} else -M[k] for k in obj]).T
            def nds(F):
                N=F.shape[0]; S=[set() for _ in range(N)]; n_=np.zeros(N,int); fronts=[[]]
                for p in range(N):
                    for q in range(N):
                        if p==q: continue
                        if np.all(F[p]<=F[q]) and np.any(F[p]<F[q]): S[p].add(q)
                        elif np.all(F[q]<=F[p]) and np.any(F[q]<F[p]): n_[p]+=1
                    if n_[p]==0: fronts[0].append(p)
                ranks=np.full(N,np.inf); i=0
                while fronts[i]:
                    nxt=[]
                    for p in fronts[i]:
                        ranks[p]=i
                        for q in S[p]:
                            n_[q]-=1
                            if n_[q]==0:nxt.append(q)
                    i+=1; fronts.append(nxt)
                return ranks,fronts[:-1]
            # evolve
            for _ in range(gens):
                M=eval_metrics(X); F=scores(M); ranks,fronts=nds(F)
                crowd=np.zeros(pop)
                for fr in fronts:
                    if len(fr)<=2: crowd[fr]=1e9; continue
                    Y=F[fr]
                    for m_ in range(Y.shape[1]):
                        o=np.argsort(Y[:,m_]); crowd[np.array(fr)[o[0]]]=crowd[np.array(fr)[o[-1]]]=1e9
                        rng_=(Y[o[-1],m_]-Y[o[0],m_]) or 1.
                        for j in range(1,len(fr)-1):
                            crowd[np.array(fr)[o[j]]]+=(Y[o[j+1],m_]-Y[o[j-1],m_])/rng_
                sel=[]
                for i in range(pop):
                    a,b=rng.choice(pop,2,replace=False)
                    sel.append(X[a] if ranks[a]<ranks[b] or (ranks[a]==ranks[b] and crowd[a]>crowd[b]) else X[b])
                kids=[]
                for j in range(0,pop,2):
                    p1,p2=sel[j],sel[(j+1)%pop]; u=rng.random(n)
                    beta=np.where(u<=0.5,(2*u)**(1/(eta_c+1)),(2*(1-u))**(-1/(eta_c+1)))
                    c1=.5*((p1+p2)-beta*(p2-p1)); c2=.5*((p1+p2)+beta*(p2-p1))
                    for c in (c1,c2):
                        if rng.random()<mut_rate:
                            u=rng.random(n); delta=np.where(u<.5,(2*u)**(1/(eta_m+1))-1,1-(2*(1-u))**(1/(eta_m+1)))
                            c+=delta
                        c=np.clip(c,lo,hi); c/=c.sum(); kids.append(c)
                X=np.array(kids)[:pop]
            # pick max Sharpe from first front
            M=eval_metrics(X); F=scores(M); _,fronts=nds(F); first=fronts[0] if fronts else range(pop)
            w_best=X[first[np.argmax(M["sharpe"][first])]]
        wL=leverage*w_best; nxt=rbd[rbd.index(t)+1] if t!=rbd[-1] else rets.index[-1]
        pr=(rets.loc[t:nxt]@wL).astype(float); pnl.loc[pr.index]=pr.values; W.append((t,wL))
    weights=pd.DataFrame({d:w for d,w in W}).T; weights.index.name="date"; weights.columns=tickers
    return {"weights":weights,"pnl":pnl.loc[weights.index[0]:],
            "details":{"method":"nsga2","objectives":obj,"pop":pop,"gens":gens,"rebalance":rebalance}}
