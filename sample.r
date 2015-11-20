## This is an R implementation of sample_dask.py. It uses the Marshall-Palmer relation
## to calculate hourly rain rates for each Id. It produces a submission file identical 
## to the 'sample_solution.csv' file provided in the Data section of this competition
## to at least 9 decimal places. It earns a MAE score of 24.06968 on the leaderboard 
## exactly equal to the sample solution benchmark. And it's faster ;)

## This may be useful in training, I'm just not sure exactly how yet. 

## Someone better in R than me could probably clean up/refactor the code a bit.


library(dplyr)
library(data.table)

test <- fread('test/test.csv')
tain<- fread('train/train.csv')


mpalmer <- function(ref, minutes_past) {
  
  # order reflectivity values and minutes_past
  sort_min_index = order(minutes_past)
  minutes_past <- minutes_past[sort_min_index]
  ref <- ref[sort_min_index]
  
  # calculate the length of time for which each reflectivity value is valid
  valid_time <- rep(0, length(minutes_past))
  valid_time[1] <- minutes_past[1]
  if (length(valid_time) > 1) {
    for (i in seq(2, length(minutes_past))) {
      valid_time[i] <- minutes_past[i] - minutes_past[i-1]
    }
    valid_time[length(valid_time)] = valid_time[length(valid_time)] + 60 - sum(valid_time)
  } else {
    # if only 1 observation, make it valid for the entire hour
    valid_time <- 60
  }
  
  valid_time = valid_time / 60
  
  # calculate hourly rain rates using marshall-palmer weighted by valid times
  sum <- 0
  for (i in seq(length(ref))) {
    if (!is.na(ref[i])) {
      mmperhr <- ((10^(ref[i]/10))/200) ^ 0.625
      sum <- sum + mmperhr * valid_time[i]
    }
  }
  
  return(sum)
  
}

results <- test %>% group_by(Id) %>% summarize(Expected=mpalmer(Ref, minutes_past))

write.csv(results, file='R_output/sample_solution.csv', row.names=FALSE)