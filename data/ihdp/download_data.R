
library(reticulate)
np <- import("numpy")
data_list <- np$load("~/DRinference/data/ihdp/ihdp_npci_1-100.train.npz")
for(i in 1:100) {

  data <- data.table(x=data_list["x"][,,i], t = as.vector(data_list["t"][,i]), y = as.vector(data_list["yf"][,i]), ate = data_list["ate"] )
  fwrite(data, paste0("./data/ihdp/", "ihdp_npci_", i, ".csv"))


}

library(reticulate)
np <- import("numpy")
data_list <- np$load("data/jobs/jobs_DW_bin.new.10.train.npz")
for(i in 1:100) {

  data <- data.table(x=data_list["x"][,,i], t = as.vector(data_list["t"][,i]), y = as.vector(data_list["yf"][,i]), ate = data_list["ate"] )
  fwrite(data, paste0("./data/jobs/", "jobs_DW_", i, ".csv"))


}
