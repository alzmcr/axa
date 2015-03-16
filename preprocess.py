import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import random
from time import time
from sklearn.linear_model import Ridge
from sklearn import feature_selection, preprocessing, ensemble, cross_validation, linear_model, metrics
from pylab import cm
from multiprocessing import Pool
import sys

driverids = os.walk('drivers').next()[1]

def driver_trips(driverid,windows=[1],n_quantiles=5,part=1,size=None):
    tripfeas = []
    windows = windows if isinstance(windows,list) else [windows]
    for i,tripid in enumerate(os.listdir('drivers/%s' % driverid)):
        trip = pd.read_csv('drivers/%s/%s' % (driverid,tripid))
        tripfea = pd.concat([trip_feature(trip,window,n_quantiles,part,size) for window in windows],axis=1)
        tripfea = tripfea.astype(np.float32)
        tripfea['driverid'] = int(driverid); tripfea['tripid'] = int(tripid.replace('.csv',''))
        tripfea['len'] = len(trip)
        tripfeas.append(tripfea.fillna(0))
    return pd.concat(tripfeas)

def trip_feature(trip,window=1,n_quantiles=5,part=1,size=None):
    diff = trip.diff(window)
    distance = (diff**2).sum(axis=1)**0.5
    acceration = distance.diff()
    heading = np.arctan2(diff.x,diff.y).diff().abs()
    cacceleration = heading*acceration
    cvelocity = heading*distance
    bounds = [((len(trip)/part)*(ip-1),(len(trip)/part)*ip) for ip in range(1,part+1)]
    if size is not None:
        size = min(len(trip)-window,size)
        # override the part
        bounds = [(window+i*size,window+(i+1)*size) for i in range((len(trip)-window)/size)]
    rows = []
    for ilower,iupper in bounds:
        # speeds
        tspeeds = [distance[ilower:iupper].quantile(q) for q in np.linspace(0,1,n_quantiles)]
        name_tspeeds = ['speed_q%.2f'%q for q in np.linspace(0,1,n_quantiles)]
        # acceleration
        tacc = [acceration[ilower:iupper].quantile(q) for q in np.linspace(0,1,n_quantiles)]
        name_tacc = ['acc_q%.2f'%q for q in np.linspace(0,1,n_quantiles)]
        #TODO heading change
        headings = [heading[ilower:iupper].quantile(q) for q in np.linspace(0,1,n_quantiles)]
        name_headings = ['headchange_q%.2f'%q for q in np.linspace(0,1,n_quantiles)]
        #TODO centripetal acceleration
        caccelerations = [cacceleration[ilower:iupper].quantile(q) for q in np.linspace(0,1,n_quantiles)]
        name_caccelerations = ['headacc_q%.2f'%q for q in np.linspace(0,1,n_quantiles)]
        #TODO as above but velocity
        cvelocitys = [cvelocity[ilower:iupper].quantile(q) for q in np.linspace(0,1,n_quantiles)]
        name_cvelocitys = ['headvel_q%.2f'%q for q in np.linspace(0,1,n_quantiles)]
        #TODO stops (inside acceleration? - (distance<1).mean())
        #TODO include standard variation in statistics?
        # append row
        rows.append([tspeeds + tacc + headings + caccelerations + cvelocitys])

    tripfea =  pd.DataFrame(
        np.array(np.vstack(rows)),
        columns = name_tspeeds + name_tacc + name_headings + name_caccelerations + name_cvelocitys
    ); tripfea.columns = ['w%i_'%window+col for col in tripfea.columns]
    return tripfea

def pool_driver_trips_handler(args):
    trip,windows,n_quantiles,part,size = args
    return driver_trips(trip,windows,n_quantiles,part,size)

def prepare_data(n_drivers, windows=[1], n_quantiles=9, part=1, size=None, n_jobs=1, seed=7):
    t = time(); random.seed(seed); n_drivers = min(n_drivers,len(driverids))
    print "Processing data | window %s | quantiles %i | (%i jobs) | ETA:" % (windows,n_quantiles,n_jobs),
    # drawing sample (if under sampling)
    sample = [ driverids[i] for i in random.sample(xrange(len(driverids)), min(n_drivers,len(driverids))) ]
    # estimated time print iteration
    eta_iteration = (np.array([2,3,4,5,10,50,100])*n_jobs).tolist() + (np.array(range(200,3000,100)*n_jobs).tolist())
    # multiprocess support
    if n_jobs > 1:
        pool = Pool(n_jobs); dfs = []; t = time()
        for i,df in enumerate(pool.imap(pool_driver_trips_handler, [(trip,windows,n_quantiles,part,size) for trip in sample])):
            dfs.append(df)
            if i in eta_iteration[4:]:
                print "%.2fm (%.2fm)," % (((time()-t) / i * (len(driverids)-i+1) / 60.),(time()-t)/60.),
                sys.stdout.flush()
        # terminate pool
        pool.terminate()
    else:
        dfs = map(lambda x: driver_trips(x,windows,n_quantiles,part), sample)
    print "DONE! %.2fm" % ((time()-t)/60.)
    return pd.concat(dfs)

if __name__ == '__main__':
    # settings
    n_drivers   = 10000
    n_jobs      = 4;    windows     = [1,15,30,60] # [1,5,10,15,30,45,60]
    part        = 6;    n_quantiles = 15
    size        = None


    fname = "data/processed_part%i_q%s_%s.csv"%(part,n_quantiles,'w'.join([str(w) for w in ['']+windows]))
    if size is not None:
        print "Using 'size'!"
        fname = "data/processed_size%i_q%s_%s.csv"%(size,n_quantiles,'w'.join([str(w) for w in ['']+windows]))

    prepare_data(n_drivers,windows,n_quantiles,part,size,n_jobs).to_csv(fname)



