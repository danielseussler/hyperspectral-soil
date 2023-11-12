library(drf)
library(tidyverse)

X_train = read.table(file = "data/interim/X_train.txt", sep = " ")
X_test = read.table(file = "data/interim/X_test.txt", sep = " ")
y_train = read.table(file = "data/interim/y_train.txt", sep = " ")


# Scale responses
y_colmeans = colMeans(y_train)
# y_train = scale(y_train, scale = FALSE)


# Fit model with standard params
fitted = drf(
  X = X_train
  , Y = y_train
  , num.trees = 10000L
  , num.features = 25
  , response.scaling = TRUE
  , seed = 23984
)


# Get predictions and scale them by training mean
preds = predict(fitted, newdata = X_test, functional = "mean")$mean
preds = preds / y_colmeans

colnames(preds) = c("P", "K", "Mg", "pH")
preds = as.data.frame(preds)

preds$id = list.files(
  "data/raw/test_data/"
  , pattern = "\\.npz$"
  , full.names = FALSE
)

preds$id = gsub("\\.npz$", "", basename(preds$id))

preds_long = preds %>%
  pivot_longer(cols = c("P", "K", "Mg", "pH"), names_to = "names", values_to = "Target") %>%
  mutate(sample_index = paste(id, "_", names, sep = "")) %>%
  select(sample_index, Target)


# get the current timestamp
timestamp = format(Sys.time(), "%Y%m%d%H%M%S")


# Create submission file
write.csv(
  x = preds_long
  , file = file.path("outputs", paste0(timestamp, "-drf.csv"))
  , row.names = FALSE
)
