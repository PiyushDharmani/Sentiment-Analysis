set.seed(15031998)
library(tm)
library(caret)
library(rminer)
library(e1071)
library(xgboost)
library(text2vec)
setwd("~/R Machine Learning Projects/TextClassificationDatasets-20190428T081607Z-001/TextClassificationDatasets/amazon_review_full_csv.tar/amazon_review_full_csv")
reviews <- read.csv("Sentiment_Analysis_Dataset_Full.csv",nrows=10000,stringsAsFactors=FALSE)

vocab <- create_vocabulary(itoken_parallel(reviews$Sentiment_Text,preprocessor = tolower,tokenizer = word_tokenizer))

dtm_train <- create_dtm(itoken_parallel(reviews$Sentiment_Text,preprocessor = tolower,tokenizer = word_tokenizer),vocab_vectorizer(vocab))

train_matrix <- xgb.DMatrix(dtm_train, label = reviews$Sentiment)
xgb_params = list(objective = "binary:logistic",eta = 0.12,max.depth = 40,eval_metric = "auc",nthread=8)
xgb_fit <- xgboost(data = train_matrix, params = xgb_params, nrounds = 10)
set.seed(1)
cv <- xgb.cv(data = train_matrix, label = reviews$Sentiment, nfold = 5,nrounds = 20,metrics = c("auc"),nthread=8)
pred <- predict(xgb_fit, dtm_train)
pred.resp <- ifelse(pred >= 0.5, 1, 0)
print(confusionMatrix(as.factor(pred.resp), as.factor(reviews$Sentiment)))










# Confusion Matrix and Statistics
# 
# Reference
# Prediction      0      1
# 0 391476  84960
# 1 105527 418037
# 
# Accuracy : 0.8095          
# 95% CI : (0.8087, 0.8103)
# No Information Rate : 0.503           
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.6189          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.7877          
#             Specificity : 0.8311          
#          Pos Pred Value : 0.8217          
#          Neg Pred Value : 0.7984          
#              Prevalence : 0.4970          
#          Detection Rate : 0.3915          
#    Detection Prevalence : 0.4764          
#       Balanced Accuracy : 0.8094          
#                                           
#        'Positive' Class : 0               
#                            
