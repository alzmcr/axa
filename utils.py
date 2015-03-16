import pandas as pd
import numpy as np

from time import sleep
from sklearn import cross_validation,metrics,cluster,decomposition,preprocessing,linear_model
from sklearn.feature_selection import f_classif
import gzip

def introduce_outliers(y,n=1,seed=10):
    y_copy = y.copy().reset_index(); np.random.seed(seed)
    y_copy['r'] = np.random.sample(len(y)); y_copy = y_copy.sort('r')
    y.loc[y_copy[-y_copy['target']][:n].set_index(['driverid','tripid']).index] = True
    return y

def low_memory_read_csv(path_fname):
    # low memory dataframe forcing numerical fields to 32 rather than 64 bits
    data_specs = pd.read_csv(path_fname,nrows=1000).dtypes.to_dict()
    for key in data_specs.keys():
        if data_specs[key] == np.int64:
            data_specs[key] = np.int32
        if data_specs[key] == np.float64:
            data_specs[key] = np.float32
    data = pd.read_csv(path_fname,dtype=data_specs)
    # remove index columns if present from pd.DataFrame.to_csv()
    if 'Unnamed: 0' in data.columns: data = data.drop('Unnamed: 0',1)
    return data

def remove_worst(submission,X,y,driverid,n_instances):
    to_remove = submission.query('driverid==%i'%driverid)[['prob']]
    index_to_remove = to_remove.sort('prob').head(n_instances).index
    return X.drop(index_to_remove),y.drop(index_to_remove)

def parse_submission(nsubmission):
    submission = pd.DataFrame.from_csv('submissions/'+nsubmission).reset_index()
    submission['driverid'] = submission['driver_trip'].apply(lambda x: int(x.split('_')[0])).values
    submission['tripid'] = submission['driver_trip'].apply(lambda x: int(x.split('_')[1])).values
    return submission.drop(['driver_trip'],1).set_index(['driverid','tripid'])

def kf_score_impf(model,X,y,n_fold=3,mask=None,seed=7):
    trscore,cvscore,impf = [],[],[]
    mask = np.array([True]*len(X)) if mask is None else mask
    for itr,icv in  cross_validation.KFold(len(X),n_fold,shuffle=True,random_state=seed):
        Xitr, Xicv = X.values[itr[mask[itr]]], X.values[icv]
        yitr, yicv = y.values[itr[mask[itr]]], y.values[icv]
        model = model.fit(Xitr,yitr)
        trscore.append( metrics.roc_auc_score(yitr, model.predict_proba(Xitr)[:,1]) )
        cvscore.append( metrics.roc_auc_score(yicv, model.predict_proba(Xicv)[:,1]) )
        impf.append( model.feature_importances_)
    return trscore, cvscore, impf

def kf_score(model,X,y,n_fold=3,mask=None,seed=7):
    trscore,cvscore = [],[]
    mask = np.array([True]*len(X)) if mask is None else mask
    for itr,icv in  cross_validation.KFold(len(X),n_fold,shuffle=True,random_state=seed):
        Xitr, Xicv = X.values[itr[mask[itr]]], X.values[icv]
        yitr, yicv = y.values[itr[mask[itr]]], y.values[icv]
        model = model.fit(Xitr,yitr)
        trscore.append( metrics.roc_auc_score(yitr, model.predict_proba(Xitr)[:,1]) )
        cvscore.append( metrics.roc_auc_score(yicv, model.predict_proba(Xicv)[:,1]) )
    return trscore, cvscore

def feature_removal(model,X,y,n_fold=3,maxiter=10,verbose=True,seed=6):
    # initial benchmark
    scoretr, scorecv, impf = kf_score_impf(model,X,y,n_fold,mask=None,seed=seed)
    cvscore_to_beat =  np.mean(scorecv)
    for i in range(maxiter):
        # feature importances
        impf = pd.Series(np.mean(impf,axis=0),X.columns); impf.sort()
        # feature p-values
        pval = pd.Series(f_classif(X,y)[0],X.columns); pval.sort()
        # select candidates for both methodologies and score removing that feature
        impf_candidate, pval_candidate = impf.index[0], pval.index[0]
        scoretr_impf, scorecv_impf, impf_impf = kf_score_impf(model,X.drop(impf_candidate,1),y,n_fold,mask=None,seed=seed)
        scoretr_pval, scorecv_pval, impf_pval = kf_score_impf(model,X.drop(pval_candidate,1),y,n_fold,mask=None,seed=seed)
        scorecv_impf, scorecv_pval = np.mean(scorecv_impf), np.mean(scorecv_pval)
        best_cvscore = max(scorecv_impf, scorecv_pval)

        if (best_cvscore - cvscore_to_beat) < 0.0005: break
        else:
            use_impf = True if best_cvscore == scorecv_impf else False
            candidate = impf_candidate if use_impf else pval_candidate
            if verbose:
                print "removing '%s' | previous %.4f | new %.4f | improvement %.4f" % (
                    candidate, cvscore_to_beat, best_cvscore, best_cvscore - cvscore_to_beat)
            cvscore_to_beat = best_cvscore
            X = X.drop(candidate,1)
    return X.columns

def feature_selection(model,X,y,n_fold=3,verbose=True,seed=11):
    feas = X.columns.tolist()
    candidates = []; fea_scores = {}; cvscore_to_beat = 0
    while True:
        # scoring on each single fea and select the best
        for fea in feas:
            fea_scores[fea] = np.mean(kf_score(model,X[candidates+[fea]],y,n_fold,mask=None,seed=seed)[1])
        best_fea = fea_scores.keys()[np.argmax(fea_scores.values())]
        best_cvscore = fea_scores[best_fea]

        # if improvement is better than the tollerance, then include fea in feature set
        if (best_cvscore - cvscore_to_beat) < 0.0005: break
        else:
            if verbose:
                print "adding '%s' | previous %.4f | new %.4f | improvement %.4f" % (
                    best_fea, cvscore_to_beat, best_cvscore, best_cvscore - cvscore_to_beat)
            candidates.append(best_fea)
            feas.remove(best_fea)
            cvscore_to_beat = best_cvscore
    return candidates

def remove_outliers(X,y,outlier_ratio=0.99,retained_var=0.99,seed=3):
    # scale at fit PCA using only the driver data point
    X_scaled = preprocessing.scale(X[y.values]); pca = decomposition.PCA().fit(X_scaled)
    # retain the selected variance and refit the PCA
    pca.n_components = np.argmax(pca.explained_variance_ratio_.cumsum() > retained_var) + 1
    # cluster using k-means (20 clusters given 200 data points to group)
    n_clusters = y.sum()/10; km = cluster.KMeans(n_clusters=n_clusters, random_state=seed)
    clusters = pd.Series(km.fit_predict(pca.fit_transform(X_scaled)))
    # smaller cluster will be considered outliers, until the sum reach the ratio desidered
    clusters_map = clusters.value_counts().cumsum() <= (len(X_scaled)*outlier_ratio)
    y_pruned = y.copy(); y_pruned[y] = clusters.map(clusters_map).values
    return y_pruned

def logger((logfile,q)):
    f = open(logfile, 'wb')
    while True:
        m = q.get()
        if m == 'kill':
            f.write('killed')
            break
        f.write(str(m) + '\n'); f.flush()
        sleep(30)
    f.close()

def create_config(logfile):
    log = pd.read_csv(logfile)
    log['feas'] = log['feas'].apply(lambda x: x.replace('[','').replace(']','').replace(' ','').replace("'",'').split(','))
    return log.set_index('driverid')

def gzip_submission(submission_pathfname):
    f_in = open(submission_pathfname, 'rb')
    f_out = gzip.open(submission_pathfname+'.gz', 'wb')
    f_out.writelines(f_in)
    f_out.close(); f_in.close()

class MyRidge(linear_model.RidgeClassifier):
    def predict_proba(self,X):
        return self._predict_proba_lr(X)

class MyRidgeCV(linear_model.RidgeClassifierCV):
    def predict_proba(self,X):
        return self._predict_proba_lr(X)
