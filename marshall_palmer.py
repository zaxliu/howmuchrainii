__author__ = 'shangxing'
#!/usr/bin/env python
#
#  Python program that will create the sample solution
#
#  @author: lak@climate.com
#  @version: $Id:$
#
#  you may have to install the master branch of 'dask' using:
#     pip install --upgrade git+https://github.com/blaze/dask
#
import dask.dataframe as dd
import pandas as pd
import numpy as np
import sys
import pickle as df


# change the location of the downloaded test file as necessary.
infile="./test.csv"
#infile="kaggle/sample.csv"
outfile="./new_solution.csv"

# Make sure you are using 64-bit python.
if sys.maxsize < 2**32:
    print("You seem to be running on a 32-bit system ... this dataset might be too large.")
else:
    print("Hurray! 64-bit.")

# read file
alldata = pd.read_csv(infile)

def marshall_palmer(ref, minutes_past):
    #print "Estimating rainfall from {0} observations".format(len(minutes_past))
    # how long is each observation valid?
    valid_time = np.zeros_like(minutes_past)
    valid_time[0] = minutes_past.iloc[0]
    for n in range(1, len(minutes_past)):
        valid_time[n] = minutes_past.iloc[n] - minutes_past.iloc[n-1]
    valid_time[-1] = valid_time[-1] + 60 - np.sum(valid_time)
    valid_time = valid_time / 60.0

    # sum up rainrate * validtime
    sum = 0
    for dbz, hours in zip(ref, valid_time):
        # See: https://en.wikipedia.org/wiki/DBZ_(meteorology)
        if np.isfinite(dbz):
            mmperhr = pow(pow(10, dbz/10)/200, 0.625)
            sum = sum + mmperhr * hours
    return sum


# each unique Id is an hour of data at some gauge
def myfunc(hour):

    hour = hour.sort('minutes_past', ascending=True)
    est = marshall_palmer(hour['Ref'], hour['minutes_past'])
    return est

estimates = alldata.groupby(alldata.Id).apply(lambda d: myfunc(d))

df = pd.DataFrame(estimates,columns=['Expected'])

df.to_csv(outfile, header=True)


