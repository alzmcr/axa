import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn import preprocessing, ensemble, cross_validation, linear_model, metrics, decomposition, cluster
from preprocess import prepare_data
from multiprocessing import Pool, Manager
from utils import logger, kf_score, kf_score_impf, feature_selection, feature_removal, remove_outliers, gzip_submission
from utils import MyRidge, MyRidgeCV, introduce_outliers, remove_worst, parse_submission, low_memory_read_csv
from time import time
import datetime
import sys

""" multiprocessing handler: assumed global data,part,qlog"""
def make_driver_prediction(driverid):
    # create mask
    np.random.seed(driverid); dmask = (data['driverid'].values==driverid)
    mask = dmask | (np.random.sample(len(data)) < (dmask.sum()*5./len(data)))
    X = data[mask].copy(); X['target'] = X['driverid'] == driverid
    X = X.set_index(['driverid','tripid']); y = X['target']; X = X.drop('target',1)

    # initialize model
    rf = ensemble.RandomForestClassifier(random_state=1, n_estimators=500, n_jobs=1, min_samples_leaf=3)

    selected_feas = X.columns; X = X[selected_feas]
    y_original = y.copy(); X_original = X.copy()
    # logging
    qlog.put('%i,"%s",%i,%i' %(driverid,selected_feas,rf.n_estimators,rf.min_samples_leaf))

    predict_with_kfold = False
    if predict_with_kfold:
        proba = y_original[y_original].copy(); proba.name = 'prob'
        for itr,icv in  cross_validation.KFold(len(X),10,shuffle=True,random_state=7):
            Xitr, Xicv = X.iloc[itr], X.iloc[icv]
            yitr, yicv, yicv_original = y.iloc[itr], y.iloc[icv], y_original.iloc[icv]
            proba[yicv[yicv_original].index.values] = rf.fit(Xitr,yitr).predict_proba(Xicv[yicv_original.values])[:,1]
    else:
        proba = pd.DataFrame(rf.fit(X,y).predict_proba(X_original[y_original])[:,1],index=X_original[y_original].index,columns=['prob'])

    return proba.reset_index()

logfile = datetime.datetime.now().strftime("logs/%Y%m%d_%H%M")+".log"

if __name__ == '__main__':
    n_jobs = 4; use_cache = False; logger = False

    if not use_cache:
        n_drivers   = 10000
        n_jobs      = 4;    windows     = [1,15,30,60]
        part        = 4;    n_quantiles = 15
        size        = None
        fname = "data/processed_part%i_q%s_%s.csv"%(part,n_quantiles,'w'.join([str(w) for w in ['']+windows]))
        data = prepare_data(n_drivers,windows,n_quantiles,part,size,n_jobs)
        data.to_csv(fname)
    else:
        # use cache
        t = time(); print "Loading cache...",
        data = pd.DataFrame.from_csv("data/processed.csv")
        print "DONE! %.2fm" % ((time()-t)/60.)

    eta_iteration = (np.array([2,3,4,5,10,50,100])*n_jobs).tolist() + (np.array(range(200,3000,100)*n_jobs).tolist())
    probas = []; t = time(); print "Predicting... estimated time:",
    if n_jobs > 1:
        # initialize logger and pool args
        qlog = Manager().Queue(); ndrivers = data['driverid'].nunique()
        # initialize pool and logger
        pool = Pool(n_jobs+(1 if logger else 0))
        if logger:
            rlog = pool.apply_async(logger, [(logfile,qlog)]); qlog.put("driverid,feas,ntree,nleaf")
        for i,proba in enumerate(pool.imap(make_driver_prediction,data['driverid'].unique()[:])):
            probas.append(proba)
            if i in eta_iteration[4:]:
                print "%.2fm (%.2fm)," % (((time()-t) / i * (data.driverid.nunique()-i+1) / 60.),(time()-t)/60.),
                sys.stdout.flush()
        qlog.put('kill'); pool.terminate()
    else:
        for i,driverid in enumerate(data['driverid'].unique()[:]):
            probas.append(make_driver_prediction(driverid))
            if i in eta_iteration:
                print "%.2fm (%.2fm)," % (((time()-t) / i * (data.driverid.nunique()-i+1) / 60.),(time()-t)/60.),
                sys.stdout.flush()
    print "DONE! %.2fm" % ((time()-t)/60.)

    print "Creating file...",
    submission_name = "submissions/%s.csv" % datetime.datetime.now().strftime("%Y%m%d_%H%M")
    submission = pd.concat(probas)[['driverid','tripid','prob']]
    submission['driver_trip'] = submission.apply(lambda x: "%i_%i"%(x['driverid'],x['tripid']),1)
    submission.groupby('driver_trip')[['prob']].mean().to_csv(submission_name, header=True)
    print "compressing..."
    gzip_submission(submission_name)
    print "DONE! %.2fm" % ((time()-t)/60.)
