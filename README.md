##AXA - Driver Telematics Analysis

### Ideas
Train N random forest model for each driver in the data set.

* Rather than consider each trip as a single instance, divide the trip
in equals part (default=4). Then sum up the probabilities at the
very end for each trip. 
* Create features with different window
* The model is train on all the driver trips
plus five times more trips randomly picked from all other drivers.

### How to generate the solution
Just run "python model.py" and you're good to go! You'll need
first to unpack the driver.zip, create a /data and /submission 
folder in order not to fail the script.

### Settings in __main__
* n_jobs: allow multiprocessing spawn N jobs
* use_cache: use previously preprocessed data file
* n_drivers: drivers to process and save to file
* windows: windows for calculate features (1,15,30,60 seconds)
* part: number of part to split a trip into
* n_quantiles: number of features to create
* size: if not None, split trip in equals size parts rather than equals parts

