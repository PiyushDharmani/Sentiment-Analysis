set.seed(15031998)
library(tm)
library(caret)
library(rminer)
library(e1071)
library(xgboost)

setwd("~/R Machine Learning Projects/TextClassificationDatasets-20190428T081607Z-001/TextClassificationDatasets/amazon_review_full_csv.tar/amazon_review_full_csv")
reviews <- read.csv("train.csv",nrows=300,stringsAsFactors=FALSE)
reviews$V2 <- paste(reviews$V2,"",reviews$V3)
reviews<- within(reviews,rm(V3))
colnames(reviews) <- c("Sentiment","Sentiment_Text")
reviews$Sentiment_Text<-gsub("[^[:alnum:] ]","",reviews$Sentiment_Text)
reviews$Sentiment_Text<-gsub("(?<=[\\s])\\s*|^\\s+|\\s+$","",reviews$Sentiment_Text, perl=TRUE)
#reviews$Sentiment_Text <- removeNumbers(reviews$Sentiment_Text)
#reviews$Sentiment_Text <- as.character(reviews$Sentiment_Text)
reviews$Sentiment <- as.numeric(reviews$Sentiment)
reviews <- reviews[!(reviews$Sentiment == 3),]
reviews$Sentiment <- replace(reviews$Sentiment,(reviews$Sentiment == 1 | reviews$Sentiment == 2 ),0)
reviews$Sentiment <- replace(reviews$Sentiment,(reviews$Sentiment == 4 | reviews$Sentiment == 5 ),1)
write.csv(reviews,file="Sentiment_Analysis_Dataset.csv",row.names = FALSE)

train_corp = VCorpus(VectorSource(reviews$Sentiment_Text))

dtm_train <- DocumentTermMatrix(train_corp, control = list(tolower = TRUE,removeNumbers = TRUE,stopwords = TRUE,removePunctuation = TRUE,stemming = TRUE))
dtm_train <- removeSparseTerms(dtm_train, 0.99)

intrain <- createDataPartition(y=reviews$Sentiment,p=0.8,list=FALSE)
dtm_train_train <- dtm_train[intrain,]
dtm_train_test <- dtm_train[-intrain,]
dtm_train_train_labels <- as.numeric(as.character(reviews[intrain,]$Sentiment))
dtm_train_test_labels <- as.factor(as.character(reviews[-intrain,]$Sentiment))

cellconvert<- function(x) {
  x <- ifelse(x > 0, "Y", "N")
}
dtm_train_train <- apply(dtm_train_train, MARGIN = 2,cellconvert)
dtm_train_test <- apply(dtm_train_test, MARGIN = 2,cellconvert)


nb_senti_classifier=naiveBayes(dtm_train_train,dtm_train_train_labels)
print("training done")
nb_predicts<-predict(nb_senti_classifier, dtm_train_test,type="class")
print(mmetric(nb_predicts, dtm_train_test_labels, c("ACC")))


