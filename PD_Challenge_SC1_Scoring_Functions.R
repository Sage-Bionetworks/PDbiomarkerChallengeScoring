# External Call:
# PD_score_challenge(training_features, test_features, walk=TRUE, leaderboard=TRUE, permute=FALSE, nperm=10000, filename="submission.csv", parentSynId="syn10695292", submissionSynId="")
# where training_features is the training matrix
# test_features is the test matrix
# walk = TRUE for walk features, FALSE for rest features
# leaderboard = TRUE for the leaderboard set, FALSE for final test set
# permute = TRUE computes permutation p-values
# nperm is the number of permutations for the p-value computation
# filename is the name of the file written to Synapse
# parentSynId is the parentId of the stored synapse file default: "syn10695292" 
# submissionSynId is the synapse Id of the submission file
#


suppressMessages(library(synapseClient))
suppressMessages(library(caret))
suppressMessages(library(caretEnsemble))
suppressMessages(library(pROC))
suppressMessages(library(glmnet))
suppressMessages(library(e1071))
suppressMessages(library(randomForest))
suppressMessages(library(kernlab))
suppressMessages(library(nnet))
suppressMessages(library(data.table))
suppressMessages(library(dplyr))

suppressMessages(synapseLogin())



PD_score_challenge<-function(submission, walk=TRUE, leaderboard=TRUE, permute=FALSE, nperm=10000, filename="submission.csv", parentSynId="syn10695292", submissionSynId=NA){
  # read in submission and create dataframes in R to avoid pandas -> R conversion issues
  pred <- read.csv(submission, header=T, as.is=T)
  split_file <- read.csv('RecordId_key_for_scoring_final.csv', header=T, as.is=T)
  test_temp <- split_file$recordId[split_file$Scored==TRUE&split_file$Training_or_Test=="Test"]
  
  train_temp <- split_file$recordId[split_file$Scored==TRUE&split_file$Training_or_Test=="Training"] #<- split_file %>%
#    filter(Training_or_Test == 'Training') %>%
#    select(-c(Scored, Training_or_Test))
#  test_features <- merge(test_temp, pred, by='recordId', all.x = TRUE)
#  training_features <- merge(train_temp, pred, by='recordId', all.x = TRUE)
  test_features<-pred[match(test_temp,pred[,1]),]
  training_features<-pred[match(train_temp,pred[,1]),]
  
  if(any(!pred[,1]%in%train_temp)){
    print("MISSING recordId(s) in Training")
  }
  if(any(!pred[,1]%in%test_temp)){
    print("MISSING recordId(s) in Test")
  }
  
  
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
  mediantraining$gender<-factor(mediantraining$gender, levels=c("Female", "Male"))
  
  dttest<-data.table(test_features)
  mediantest<-dttest[, lapply(.SD, median, na.rm=TRUE), by=groupvariables, .SDcols=featurenames ]
  mediantest<-as.data.frame(mediantest)
  mediantest$gender<-factor(mediantest$gender, levels=c("Female", "Male"))
  
  
  print("Fitting the model")
  #Fit model
  ensemble_model<-fit_model(mediantraining, featurenames, covs_num, covs_fac)
  
  print("Scoring the model")
  #Score_model
  final_score<-score_model(mediantest, ensemble_model, featurenames, covs_num, covs_fac, permute, nperm)
  print(final_score$scores)
  
  suppressWarnings(write.csv(final_score$predictions, filename, row.names=F, quote=F))
  syn<-File(filename, parentId=parentSynId)
  print(syn) # remove
  
  if(!is.na(submissionSynId)){
    syn<-synStore(syn, used=submissionSynId, 
                executed='https://github.com/Sage-Bionetworks/PDbiomarkerChallengeScoring/blob/master/PD_Challenge_SC1_Scoring_Functions.R',
                activityName = "Subchallenge 1 Scoring",
                activityDescription = "Fit ensemble learning on training and predict on test.")
  } else {
    syn<-synStore(syn,
                  executed='https://github.com/Sage-Bionetworks/PDbiomarkerChallengeScoring/blob/master/PD_Challenge_SC1_Scoring_Functions.R',
                  activityName = "Subchallenge 1 Scoring",
                  activityDescription = "Fit ensemble learning on training and predict on test.")
  }
  file.remove(filename)
  
  auroc <- final_score$scores['AUROC']
  pval <- final_score$scores['pval']
  
  return(list(auroc, pval))
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


