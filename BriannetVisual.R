library(readxl)
library(readr)
library(reticulate)
library(brainconn)
getwd()
setwd("F:/projects/AiyingT1MultimodalFusion/notebooks")
CAB_NP_v1_1_Labels_ReorderedbyNetworks <- read_excel("CAB-NP_v1.1_Labels-ReorderedbyNetworks.xlsx")


glassermni <- read_csv("glassermni.txt")

m_g <- dim(glassermni)[1]

glasser <- read_table("glasser.node", col_names = FALSE)



glassertemp <- data.frame(glassermni$regionName,as.integer(round(glasser$X1)),as.integer(round(glasser$X2)),as.integer(round(glasser$X3)),CAB_NP_v1_1_Labels_ReorderedbyNetworks$NETWORK[1:m_g])


colnames(glassertemp) <-  c('ROI.Name','x.mni','y.mni','z.mni','network')


check_atlas(glassertemp)

np <- import("numpy")
conmat <- np$load("mask_reg.npy")
print(class(conmat))
print(dim(conmat))

conmat_r <- matrix(as.numeric(np$array(conmat)), nrow = 360, ncol = 360, byrow = TRUE)
brainconn(atlas = glassertemp, conmat = conmat_r)

# ## Example
# conmat <- read.table("km_glasser_1_male.edge", header = FALSE, as.is = TRUE)
# 
# # Verify that conmat is numeric
# # If conmat is a data frame, convert it to a matrix
# if (is.data.frame(conmat)) {
#   conmat <- as.matrix(conmat)
# }
# 
# # Ensure all elements are numeric
# if (!all(sapply(conmat, is.numeric))) {
#   conmat <- apply(conmat, 2, as.numeric)
# }
# 
# # Now pass the numeric matrix to brainconn
# brainconn(atlas = glassertemp, conmat = conmat)

