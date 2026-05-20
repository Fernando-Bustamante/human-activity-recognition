rm(list = ls())

# Loading Data
X_train  <- read.table("UCI HAR Dataset/train/X_train.txt")
y_train  <- read.table("UCI HAR Dataset/train/y_train.txt")[[1]]
features <- read.table("UCI HAR Dataset/features.txt")[[2]]
colnames(X_train) <- features

act_file   <- read.table("UCI HAR Dataset/activity_labels.txt")
activities <- act_file[[2]]
y_label    <- factor(y_train, labels = activities)

# F-score
fscore <- sapply(1:ncol(X_train), function(j) {
  summary(aov(X_train[, j] ~ y_label))[[1]][["F value"]][1]
})
names(fscore) <- colnames(X_train)

# Best F-score, skipping correlated variables (>0.9)
ranked <- names(sort(fscore, decreasing = TRUE))
top20 <- character(0)
for (v in ranked) {
  if (length(top20) == 0 || max(abs(cor(X_train[, v], X_train[, top20]))) < 0.9) {
    top20 <- c(top20, v)
  }
  if (length(top20) == 20) break
}

# List of the top 20 selected variables
cat("Top 20 selected variables:\n")
print(top20)

data <- X_train[, top20]
data$Activity <- y_label

library(nnet)
colnames(data) <- make.names(colnames(data))

# 5-fold cross-validation
set.seed(42)
k <- 5
folds <- sample(rep(1:k, length.out = nrow(data)))  # assign each row to a fold
train_acc <- numeric(k)
test_acc  <- numeric(k)
for (i in 1:k) {
  train_set <- data[folds != i, ]   # 4 folds for training
  test_set  <- data[folds == i, ]   # 1 fold for testing
  m <- nnet(Activity ~ ., data = train_set, size = 12, maxit = 300, trace = FALSE)

  p_train <- predict(m, train_set, type = "class")
  p_test  <- predict(m, test_set,  type = "class")
  train_acc[i] <- mean(p_train == train_set$Activity)
  test_acc[i]  <- mean(p_test  == test_set$Activity)
}
cat("Train accuracy per fold:", round(train_acc, 4), "\n")
cat("Test accuracy per fold: ", round(test_acc, 4), "\n")
cat("Mean train:", round(mean(train_acc), 4), "\n")
cat("Mean test: ", round(mean(test_acc), 4), "± sd:", round(sd(test_acc), 4), "\n")

# Confusion matrix of the last fold (in the Plots pane, without saving PNG)
cm <- table(Predicted = p_test, Real = test_set$Activity)

image(1:ncol(cm), 1:nrow(cm), t(cm[nrow(cm):1, ]),
      col = hcl.colors(20, "Blues", rev = TRUE),
      axes = FALSE, xlab = "Real", ylab = "Predicted",
      main = "Confusion Matrix")
axis(1, 1:ncol(cm), colnames(cm), las = 2, cex.axis = 0.7)
axis(2, 1:nrow(cm), rev(rownames(cm)), las = 2, cex.axis = 0.7)
for (i in 1:nrow(cm))
  for (j in 1:ncol(cm))
    text(j, nrow(cm) - i + 1, cm[i, j])