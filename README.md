# Human Activity Recognition using Smartphones

Classification of human physical activities using neural networks and intelligent feature selection applied to the UCI HAR dataset.

## Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Samples:** 7,352 training / 2,947 test
- **Features:** 561 variables extracted from accelerometer and gyroscope signals
- **Classes:** 6 activities — WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING

## Methodology

### 1. Feature Selection

With 561 variables, selecting the most relevant ones is essential. A two-step greedy selection was applied:

**Step 1 — F-score (ANOVA):** ranks each variable by how well it separates the 6 activity classes.

```
F = between-class variance / within-class variance
```

The higher the F, the better the variable discriminates between activities.

**Step 2 — Correlation filter:** iterates through variables ranked by F-score and only adds a variable to the final set if its correlation with all already-selected variables is below 0.9. This eliminates redundancy and ensures complementary information.

This greedy approach combines relevance and non-redundancy, inspired by the mRMR criterion.

### 2. Neural Network (MLP)

- **Package:** `nnet` (R)
- **Architecture:** 20 inputs → 12 hidden neurons (sigmoid) → 6 outputs (softmax)
- **Optimizer:** BFGS (quasi-Newton method)
- **Evaluation:** 5-fold cross-validation

## Results

| Metric | Value |
|---|---|
| Mean train accuracy | 98.64% |
| Mean test accuracy | **95.50%** |
| Std deviation (test) | ± 0.68% |

**95.5% accuracy using only 20 out of 561 variables — 3.5% of the original features.**

For reference, the original UCI HAR paper reports ~96% using all 561 variables.

### Top 20 Selected Variables

| # | Variable | Domain |
|---|---|---|
| 1 | fBodyAccJerk-entropy()-X | Frequency |
| 2 | tGravityAcc-mean()-X | Time |
| 3 | tBodyAccMag-energy() | Time |
| 4 | fBodyAcc-max()-Y | Frequency |
| 5 | fBodyGyro-std()-Z | Frequency |
| 6 | tBodyGyro-iqr()-X | Time |
| 7 | tBodyAcc-iqr()-Z | Time |
| 8 | fBodyAccJerk-mean()-Z | Frequency |
| 9 | tBodyGyroJerk-max()-Z | Time |
| 10 | tGravityAcc-max()-Y | Time |
| 11 | fBodyGyro-iqr()-X | Frequency |
| 12 | fBodyBodyGyroMag-iqr() | Frequency |
| 13 | tBodyAcc-entropy()-X | Time |
| 14 | tBodyAcc-entropy()-Y | Time |
| 15 | fBodyBodyGyroMag-std() | Frequency |
| 16 | tBodyGyro-iqr()-Y | Time |
| 17 | tBodyGyro-min()-X | Time |
| 18 | tBodyGyro-max()-X | Time |
| 19 | tBodyAccJerk-energy()-Y | Time |
| 20 | tBodyAccMag-min() | Time |

## How to Run

**Requirements:** R with packages `nnet` and `corrplot`

```r
install.packages(c("nnet", "corrplot"))
```

1. Download the dataset from the UCI link above and extract to `UCI HAR Dataset/`
2. Open `HAR.r` in RStudio or VS Code
3. Set the working directory to this folder
4. Run the script

## File Structure

```
├── HAR.r                      # Main script: feature selection + MLP + cross-validation
├── HAR_top20_variaveis.png    # Correlation matrix of top 20 selected variables
├── .gitignore
└── README.md
```
