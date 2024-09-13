
set.seed(1)
library(naniar)
library(xgboost)
library(MLmetrics)
library(caret)
library(randomForest)
library(xtable)
library(shapviz)
library(ggplot2)
library(gridExtra)
library(ggpubr)


# DATA CLEANING


variables_ph10 <- c(
  "RHISPANIC",
  "RRACE",
  "EEDUC",  
  "MS",
  "THHLD_NUMPER",
  "THHLD_NUMKID",
  "WRKLOSSRV",
  "EST_ST",
  "TENURE",
  "INCOME",
  "HLTH_MHCHLD1",
  "HLTH_MHCHLD2",
  "HLTH_MHCHLD9",
  "HWEIGHT"
)
variables_ph5to8 <- c(
  "RHISPANIC",
  "RRACE",
  "EEDUC",  
  "MS",
  "THHLD_NUMPER",
  "THHLD_NUMKID",
  "WRKLOSSRV",
  "EST_ST",
  "TENURE",
  "INCOME",
  "KIDBHVR1",
  "KIDBHVR2",
  "KIDBHVR9",
  "HWEIGHT"
)

# loop through weeks for phase 3.10
start_week_ph10 <- 61
end_week_ph10 <- 63
data_ph10 <- data.frame(matrix(ncol = length(variables_ph10), nrow = 0))
for (i in start_week_ph10:end_week_ph10) {
  data_file <- paste0('HPS_Week', i, '_PUF_CSV/pulse2023_puf_', i, '.csv')
  data_curr <- read.csv(file = data_file, header = TRUE)
  data_curr <- data_curr[, variables_ph10]
  data_ph10 <- rbind(data_ph10, data_curr)
}

# loop through weeks for phase 3.5 to 3.8
start_week_ph5to8 <- 46
end_week_ph5to8 <- 57
data_ph5to8 <- data.frame(matrix(ncol = length(variables_ph5to8), nrow = 0))

for (i in start_week_ph5to8:end_week_ph5to8) {
  if (i <= 52) {
    data_file <- paste0('HPS_Week', i, '_PUF_CSV/pulse2022_puf_', i, '.csv')
  }
  else {
    data_file <- paste0('HPS_Week', i, '_PUF_CSV/pulse2023_puf_', i, '.csv')
  }
  data_curr <- read.csv(file = data_file, header = TRUE)
  data_curr <- data_curr[, variables_ph5to8]
  data_ph5to8 <- rbind(data_ph5to8, data_curr)
}

names(data_ph5to8) <- names(data_ph10)
df_raw <- rbind(data_ph5to8, data_ph10)
df <- df_raw[df_raw$THHLD_NUMKID > 0,]
dim(df)


# define response
df$KidMentalIssue <- 1 * ((df$HLTH_MHCHLD1 > 0 | df$HLTH_MHCHLD2 > 0) & df$HLTH_MHCHLD9 < 0) + 
  (-88) * ((df$HLTH_MHCHLD1 < 0 & df$HLTH_MHCHLD2 < 0 & df$HLTH_MHCHLD9 < 0) | ((df$HLTH_MHCHLD1 > 0 | df$HLTH_MHCHLD2 > 0) & df$HLTH_MHCHLD9 > 0))

df <- replace_with_na(df,
                      replace = list(KidMentalIssue = -88))
df$KidMentalIssue <- factor(df$KidMentalIssue, 
                            labels = c('0', '1'))
table(df$KidMentalIssue)


# preprocessing covariates
df$RHISPANIC <- factor(df$RHISPANIC,
                       labels = c('NotHisp', 'Hispanic'))

df$RRACE <- factor(df$RRACE,
                   labels = c('White', 'Black', 'Asian', 'Others'))
df$RACE <- factor( 
  1 * (df$RHISPANIC == 'NotHisp' & df$RRACE == 'White') + 
    2 * (df$RHISPANIC == 'NotHisp' & df$RRACE == 'Black') + 
    3 * (df$RHISPANIC == 'NotHisp' & df$RRACE == 'Asian') + 
    4 * (df$RHISPANIC == 'Hispanic') +
    5 * (df$RHISPANIC == 'NotHisp' & df$RRACE == 'Others'),
  labels = c('NotHispWhite', 'NotHispBlack', 'NotHispAsian', 'Hispanic', 'Others'))
df$EEDUC <- factor(
  df$EEDUC,
  labels = c('Less_than_high_school', 'Some_high_school', 'High_school_graduate',
             'Some_college', 'Associate', 'Bachelor', 'Graduate'))
df$KIDPROP <- df$THHLD_NUMKID/df$THHLD_NUMPER
df$EST_ST <- factor(df$EST_ST, 
                    labels = c('Alabama', 'Alaska', 'Arizona', 'Arkansas', 
                               'California', 'Colorado', 'Connecticut',
                               'Delaware', 'District_of_Columbia',
                               'Florida',
                               'Georgia',
                               'Hawaii',
                               'Idaho', 'Illinois', 'Indiana', 'Iowa',
                               'Kansas', 'Kentucky',
                               'Louisiana',
                               'Maine', 'Maryland', 'Massachusetts', 'Michigan',
                               'Minnesota', 'Mississippi', 'Missouri', 'Montana',
                               'Nebraska', 'Nevada', 'New_Hampshire', 'New_Jersey',
                               'New_Mexico', 'New_York', 'North_Carolina', 'North_Dakota',
                               'Ohio', 'Oklahoma', 'Oregon',
                               'Pennsylvania',
                               'Rhode_Island',
                               'South_Carolina', 'South_Dakota',
                               'Tennessee', 'Texas',
                               'Utah',
                               'Vermont', 'Virginia',
                               'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming'))

df <- replace_with_na(df,
                      replace = list(WRKLOSSRV = c(-88, -99), 
                                     TENURE = c(-88, -99),
                                     INCOME = c(-88, -99),
                                     MS = -99))

df$WRKLOSSRV <- factor(df$WRKLOSSRV,
                       labels = c('Yes', 'No'))
df$MS <- factor(
  df$MS,
  labels = c('Now_Married', 'Widowed', 'Divorced', 'Separated', 'Never_Married')
)

df$INCOME <- factor(
  df$INCOME,
  labels = c('less_than_25K', '25K_35K', '35K_50K', '50K_75K', '75K_100K',
             '100K_150K', '150K_200K', 'more_than_200K')
)

df$TENURE <- factor(
  df$TENURE,
  labels = c('Owned', 'Loan', 'Rented', 'Occupied')
)

# re-select covariates
df <- df[, c('KidMentalIssue', 'KIDPROP', 'RACE', 'EEDUC', 
             'MS', 'EST_ST', 'WRKLOSSRV', 'TENURE', 'INCOME', 'HWEIGHT')]


# remove missing value for now
df <- na.omit(df)
hweight <- df$HWEIGHT
df <- df[, c('KidMentalIssue', 'KIDPROP', 'RACE', 'EEDUC', 
             'MS', 'EST_ST', 'WRKLOSSRV', 'TENURE', 'INCOME')]

dim(df)
head(df)


#write.csv(df, "hps_cleaned.csv")


# XGB MODEL


# convert dataframe into xgb.DMatrix
# split data into training and test data
df.xgb <- xgb.DMatrix(data.matrix(df[, -1]), 
                      label = as.numeric(df$KidMentalIssue) - 1 , nthread = 1)

train_index <- sample(1:nrow(df), round(0.7*nrow(df)))
df_train <- df[train_index, ]
df_test <- df[-train_index, ]

xgb_train <- xgb.DMatrix(data.matrix(df_train[, -1]), 
                         label = as.numeric(df_train$KidMentalIssue) - 1 , 
                         nthread = 1)
xgb_test <- xgb.DMatrix(data.matrix(df_test[, -1]), 
                        label = as.numeric(df_test$KidMentalIssue) - 1 , 
                        nthread = 1)
# XGBoost tuning
f1 <- function(data, lev = NULL, model = NULL) {
  f1_val <- MLmetrics::F1_Score(y_pred = data$pred,
                                y_true = data$obs)
  c(F1 = f1_val)
}

train.control <- trainControl(method = "cv", number = 5, search = "random", 
                              summaryFunction = f1)

xgb.tuning.l <- train(KidMentalIssue~., data = df_train, method = "xgbTree",
                      objective = "binary:logistic", maximize = TRUE,
                      trControl = train.control, metric = "F1")

xgb.tuning.h <- train(KidMentalIssue~., data = df_train, method = "xgbTree",
                      objective = "binary:hinge", maximize = TRUE,
                      trControl = train.control, metric = "F1")

# XGBoost model fitting
xgb.logistic <- xgboost(learning_rate = xgb.tuning.l$bestTune[1, "eta"], 
                        nthread = 1, verbose = 0, 
                        data = xgb_train, nrounds = xgb.tuning.l$bestTune[1, "nrounds"], 
                        gamma = xgb.tuning.l$bestTune[1, "gamma"],
                        objective = "binary:logistic", booster = "gbtree", 
                        max_depth = xgb.tuning.l$bestTune[1, "max_depth"], 
                        colsample_bytree = xgb.tuning.l$bestTune[1, "colsample_bytree"], 
                        min_child_weight = xgb.tuning.l$bestTune[1, "min_child_weight"], 
                        subsample = xgb.tuning.l$bestTune[1, "subsample"])
xgb.logistic.probs <- predict(xgb.logistic, xgb_test, type = 'response')
xgb.logistic.pred <- rep(0, nrow(df_test))
xgb.logistic.pred[xgb.logistic.probs > 0.5] <- 1

# best tuning here overfitting
#xgb.hinge <- xgboost(learning_rate = xgb.tuning.h$bestTune[1, "eta"], 
#                     nthread = 1, verbose = 0, 
#                     data = xgb_train, nrounds = xgb.tuning.h$bestTune[1, "nrounds"], 
#                     gamma = xgb.tuning.h$bestTune[1, "gamma"],
#                     objective = "binary:hinge", booster = "gbtree", 
#                     max_depth = xgb.tuning.h$bestTune[1, "max_depth"], 
#                     colsample_bytree = xgb.tuning.h$bestTune[1, "colsample_bytree"], 
#                     min_child_weight = xgb.tuning.h$bestTune[1, "min_child_weight"], 
#                     subsample = xgb.tuning.h$bestTune[1, "subsample"])

xgb.hinge <- xgboost(learning_rate = xgb.tuning.h$results[1, "eta"], 
                     nthread = 1, verbose = 0, 
                     data = xgb_train, nrounds = xgb.tuning.h$results[1, "nrounds"], 
                     gamma = xgb.tuning.h$results[1, "gamma"],
                     objective = "binary:hinge", booster = "gbtree", 
                     max_depth = xgb.tuning.h$results[1, "max_depth"], 
                     colsample_bytree = xgb.tuning.h$results[1, "colsample_bytree"], 
                     min_child_weight = xgb.tuning.h$results[1, "min_child_weight"], 
                     subsample = xgb.tuning.h$results[1, "subsample"])
xgb.hinge.pred <- predict(xgb.hinge, xgb_test)


# OTHER MODELS


# logistic regression
model.logistic <- glm(KidMentalIssue ~., data = df_train, family = binomial)
model.logistic.probs <- predict(model.logistic, df_test[, -1], type = 'response')
model.logistic.pred <- rep(0, nrow(df_test))
model.logistic.pred[model.logistic.probs > 0.5] <- 1

# random forest
start.time <- Sys.time()
model.rf <- randomForest(KidMentalIssue ~. , data = df_train, 
                         mtry = round(sqrt(ncol(df_train))), importance = T,
                         ntree = 500)
end.time <- Sys.time()
round(end.time - start.time, 2)
time.taken
model.rf.pred <- predict(model.rf, df_test[, -1], type = 'response')


# EVALUATION
# F1 
ls.f1 <- c(F1_Score(df_test[, 1], xgb.logistic.pred), 
           F1_Score(df_test[, 1], xgb.hinge.pred),
           F1_Score(df_test[, 1], model.logistic.pred),
           F1_Score(df_test[, 1], model.rf.pred))

# AUC
ls.auc <- c(AUC(df_test[, 1], xgb.logistic.pred), 
            AUC(df_test[, 1], xgb.hinge.pred), 
            AUC(df_test[, 1], model.logistic.pred),
            AUC(df_test[, 1], model.rf.pred))

# ACC
ls.acc <- c(mean(df_test[, 1] == xgb.logistic.pred), 
            mean(df_test[, 1] == xgb.hinge.pred),
            mean(df_test[, 1] == model.logistic.pred), 
            mean(df_test[, 1] == model.rf.pred))

# print table
df.evaluation <- data.frame(ls.f1, ls.auc, ls.acc)
names(df.evaluation) <- c("F1", "AUC", "ACC")
rownames(df.evaluation) <- c("XGBoost: Logistic Loss", "XGBoost: Hinge Loss",
                             "Logistic Regression", "Random Forest")


xtable(df.evaluation, digits = 5, align = c("l", "l", "l", "l"))


# SHAP


# mean absolute SHAP values
## prepare models for SHAP visualization
shap.logistic <- shapviz(xgb.logistic, X_pred = data.matrix(df_test[, -1]), X = df_test)
shap.hinge <- shapviz(xgb.hinge, X_pred = data.matrix(df_test[, -1]), X = df_test)
## plots
p.logistic.m <- sv_importance(shap.logistic) + coord_flip() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7), 
        plot.title = element_text(size = 10, hjust = 0.5),
        axis.title.y = element_text(size = 10)) + 
  labs(x = "Mean Absolute SHAP Values", y = "") +
  ggtitle("Logistic Loss")
p.hinge.m <- sv_importance(shap.hinge) + coord_flip() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        plot.title = element_text(size = 10, hjust = 0.5)) +
  labs(y = "", x = "") +
  ggtitle("Hinge Loss")
grid.arrange(p.logistic.m, p.hinge.m, ncol = 2)


# beeswarm plot
bee.logistic <- sv_importance(shap.logistic, kind = "beeswarm") +
  theme(plot.title = element_text(size = 10, hjust = 0.5),
        axis.title.x = element_text(size = 10)) + 
  labs(x = "SHAP Values", y = "") +
  ggtitle("Logistic Loss")
bee.hinge <- sv_importance(shap.hinge, kind = "beeswarm") + 
  theme(plot.title = element_text(size = 10, hjust = 0.5),
        axis.title.x = element_text(size = 10)) + 
  labs(x = "SHAP Values", y = "") +
  ggtitle("Hinge Loss")
grid.arrange(bee.logistic, bee.hinge, nrow = 2)

# individual SHAP values barplot, ID = 1
shap1.logistic <- shapviz(xgb.logistic, X_pred = data.matrix(df_test[1, -1]), X = df_test[1, ])
shap1.hinge <- shapviz(xgb.hinge, X_pred = data.matrix(df_test[1, -1]), X = df_test[1, ])

p.logistic1 <- sv_importance(shap1.logistic) + coord_flip() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7), 
        plot.title = element_text(size = 10, hjust = 0.5),
        axis.title.y = element_text(size = 10)) + 
  labs(x = "Absolute SHAP Values", y = "") +
  ggtitle("Logistic Loss")
p.hinge1 <- sv_importance(shap1.hinge) + coord_flip() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
        plot.title = element_text(size = 10, hjust = 0.5)) +
  labs(y = "", x = "") +
  ggtitle("Hinge Loss")
grid.arrange(p.logistic1, p.hinge1, ncol = 2)


# individual SHAP values waterplot, ID = 1
sv_waterfall(shap1.logistic) + 
  theme(axis.title.y = element_text(size = 7)) +
  labs(x = "SHAP Values", y = "") 
sv_waterfall(shap1.hinge) + 
  theme(axis.title.y = element_text(size = 7)) +
  labs(x = "SHAP Values", y = "") 


# dependence plots
## section 1: first three variables
## KIDPROP
p.kp.l <- sv_dependence(shap.logistic, "KIDPROP", color_var = NULL, size = 0.5) +
  theme(plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") +
  ggtitle("KIDPROP") 
p.kp.h <- sv_dependence(shap.hinge, "KIDPROP", color_var = NULL, size = 0.5) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") +
  theme(plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  ggtitle("KIDPROP") 
## RACE
p.r.l <- sv_dependence(shap.logistic, "RACE", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") +
  ggtitle("RACE")
p.r.h <- sv_dependence(shap.hinge, "RACE", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") +
  ggtitle("RACE")
## EEDUC
p.ed.l <- sv_dependence(shap.logistic, "EEDUC", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") +
  ggtitle("EEDUC")
p.ed.h <- sv_dependence(shap.hinge, "EEDUC", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") +
  ggtitle("EEDUC")

## section 2: EST_ST
p.est.l <- sv_dependence(shap.logistic, "EST_ST", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) + 
  labs(y = "SHAP Values: \n Logistic Loss", x = "") +
  ggtitle("EST_ST") 
p.est.h <- sv_dependence(shap.hinge, "EST_ST", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "")
## combined plots
grid.arrange(p.est.l, p.est.h, nrow = 2)


# section 3: last four variables
## MS
p.ms.l <- sv_dependence(shap.logistic, "MS", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") +
  ggtitle("MS")
p.ms.h <- sv_dependence(shap.hinge, "MS", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") +
  ggtitle("MS")
## WRKLOSSRV
p.wl.l <- sv_dependence(shap.logistic, "WRKLOSSRV", color_var = NULL, size = 0.5) +
  theme(plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "")+ 
  ggtitle("WRKLOSSRV")
p.wl.h <- sv_dependence(shap.hinge, "WRKLOSSRV", color_var = NULL, size = 0.5)+
  theme(plot.title = element_text(size = 7, hjust = 0.5),
        axis.title.y = element_text(size = 7)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") + 
  ggtitle("WRKLOSSRV")
## TENURE
p.tn.l <- sv_dependence(shap.logistic, "TENURE", color_var = NULL, size = 0.5)  +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") + 
  ggtitle("TENURE")
p.tn.h <- sv_dependence(shap.hinge, "TENURE", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") + 
  ggtitle("TENURE")
## INCOME
p.ic.l <- sv_dependence(shap.logistic, "INCOME", color_var = NULL, size = 0.5) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Logistic Loss", x = "") + 
  ggtitle("INCOME")
p.ic.h <- sv_dependence(shap.hinge, "INCOME", color_var = NULL, size = 0.5)+
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 5),
        axis.title.y = element_text(size = 7),
        plot.title = element_text(size = 7, hjust = 0.5)) +
  labs(y = "SHAP Values: \n Hinge Loss", x = "") + 
  ggtitle("INCOME")
## combined plots


ggarrange(p.kp.l, p.kp.h, p.r.l, p.r.h, 
          p.ed.l, p.ed.h, p.ms.l, p.ms.h, 
          p.wl.l, p.wl.h, p.tn.l, p.tn.h, 
          p.ic.l, p.ic.h, nrow = 4, ncol = 4)

