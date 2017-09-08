# External Call:
# PD_score_challenge(training_features, test_features, walk=TRUE, leaderboard=TRUE)
# where training_features is the training matrix
# test_features is the test matrix
# walk = TRUE for walk features, FALSE for rest features
# leaderboard = TRUE for the leaderboard set, FALSE for final test set
#

library(synapseClient)
library(caret)
library(caretEnsemble)
library(pROC)
library(glmnet)
library(e1071)
library(randomForest)
library(kernlab)
library(nnet)
library(data.table)

synapseLogin()



PD_score_challenge<-function(training_features, test_features, walk=TRUE, leaderboard=TRUE, permute=FALSE, nperm=10000){
  #manipulate incoming data into dataframe
  training_features<-as.data.frame(training_features)
  test_features<-as.data.frame(test_features)
  recordidname<-names(training_features)[1]
  featurenames<-names(training_features)[-1]
  training_features[,featurenames] <- sapply( training_features[,featurenames], as.numeric )
  test_features[,featurenames] <- sapply( test_features[,featurenames], as.numeric )
  
  print("Reading and merging covariates")
  #Read-in and merge covariates
  training_features<-download_merge_covariate(training_features, walk=walk, leaderboard=leaderboard, training=TRUE)
  test_features<-download_merge_covariate(test_features, walk=walk, leaderboard=leaderboard, training=FALSE)
  
  print("Summarizing the data")
  #Summarize by Median
  covs_num<-c("age")
  covs_fac<-c("gender")
  groupvariables<-c("healthCode", "medTimepoint", "professional.diagnosis", covs_num, covs_fac) #"age", "gender") #, "appVersion.walktest", "phoneInfo.walktest")
  
  dttrain<-data.table(training_features)
  mediantraining<-dttrain[, lapply(.SD, median, na.rm=TRUE), by=groupvariables, .SDcols=featurenames ]
  mediantraining<-as.data.frame(mediantraining)
  
  dttest<-data.table(test_features)
  mediantest<-dttest[, lapply(.SD, median, na.rm=TRUE), by=groupvariables, .SDcols=featurenames ]
  mediantest<-as.data.frame(mediantest)
  
  print("Fitting the model")
  #Fit model
  ensemble_model<-fit_model(mediantraining, featurenames, covs_num, covs_fac)
  
  print("Scoring the model")
  #Score_model
  final_score<-score_model(mediantest, ensemble_model, featurenames, covs_num, covs_fac, permute, nperm)
  
  return(final_score)
}



download_merge_covariate<-function(features, walk=TRUE, leaderboard=FALSE, training=TRUE){
  
  #download correct covariates
  if(walk){
    if(training){
      synid<-"syn10233116"
    } else if(leaderboard) {
      synid<-"syn10233119"
    } else {
      synid<-"syn10233122"
    }
  } else {
    if(training){
      synid<-"syn10233132"
    } else if(leaderboard) {
      synid<-"syn10233139"
    } else {
      synid<-"syn10233142"
    }    
  }
  
  syndemos<-synGet(synid)
  demos<-read.csv(attributes(syndemos)$filePath, header=T, as.is=T)
  
  
  # Merge Data
  if(walk){
    mergeddata<-cbind(demos, features[match(demos$recordId.walktest, features[,1]), -1])
  } else {
    mergeddata<-cbind(demos, features[match(demos$recordId.resttest, features[,1]), -1])
  }
  return(mergeddata)
}

fit_model<-function(training, featurenames, covs_num, covs_fac){
  trainoutcome<-training$professional.diagnosis
  trainoutcome<-factor(ifelse(trainoutcome, "PD", "Control"))
  
  trainfeatures<-training[,c(covs_num, covs_fac,featurenames)]
  trainfeatures[,covs_fac]<-sapply(trainfeatures[,covs_fac], as.factor)

  print("Run Control")  
  set.seed(1234)
  myControl <- trainControl(  method="boot",
                              number = 50,
                              savePredictions="final",
                              classProbs=TRUE,
                              index=createResample(trainoutcome, 50),
                              summaryFunction=twoClassSummary )
  
  
  tmpmmatrix<-model.matrix( ~ ., trainfeatures)
  tmpmmatrix<-tmpmmatrix[,-1]
  
  print(dim(tmpmmatrix))
  print(length(trainoutcome))
  print(table(trainoutcome))
  
  print("Caret Call")
  model_list <- caretList(
    y=trainoutcome , x=tmpmmatrix,
    trControl=myControl,
    metric = "ROC",
    tuneList=list(glmnet = caretModelSpec(method = "glmnet"), 
                  rf = caretModelSpec(method="rf"), 
                  svmLinear = caretModelSpec(method="svmLinear"), 
                  knn = caretModelSpec(method="knn"), 
                  nnet = caretModelSpec(method = "nnet", trace=F)),
    continue_on_fail = FALSE, preProc = c("center", "scale")
  )
  
  print("Ensemble run")
  greedy_ensemble <- caretEnsemble(
    model_list, 
    metric="ROC",
    trControl=trainControl(
      number=length(model_list),
      summaryFunction=twoClassSummary,
      classProbs=TRUE
    ))
  return(greedy_ensemble)
}

score_model<-function(testing, ensemble_model, featurenames, covs_num, covs_fac, permute, nperm){
  testoutcome<-testing$professional.diagnosis
  testoutcome<-factor(ifelse(as.logical(testoutcome), "PD", "Control"))
  
  testfeatures<-testing[,c(covs_num, covs_fac,featurenames)]
  testfeatures[,covs_fac]<-sapply(testfeatures[,covs_fac], as.factor)
  
  tmpmmatrix<-model.matrix( ~ ., testfeatures)
  tmpmmatrix<-tmpmmatrix[,-1]
  
#  return(cbind(testoutcome, tmpmmatrix))
  print(dim(tmpmmatrix))
  print(length(testoutcome))
  print(table(testoutcome))
  
  
  preds_greedensemble <- predict(object=ensemble_model, tmpmmatrix, type="prob")
  score<-roc(ifelse(testoutcome=="PD",1,0), preds_greedensemble)$auc
  predframe<-data.frame(healthCode=testing$healthCode, Dx=testoutcome, Prediction=preds_greedensemble, stringsAsFactors = FALSE)
  
  perm=NA
  if(permute){
    permauc<-replicate(nperm, roc(ifelse(testoutcome=="PD",1,0), sample(preds_greedensemble))$auc)
    perm<-(sum(score>=permauc)+1)/(nperm+1)
  }
  res<-list(scores=c(AUROC=score, pval=perm), predictions=predframe)
  return(res)
}



  
  


