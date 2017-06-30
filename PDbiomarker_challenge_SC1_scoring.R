
library(caret)
library(caretEnsemble)
#library(mlbench)
library(pROC)
library(synapseClient)

library(CovariateAnalysis) # get it from #devtools::install_github('th1vairam/CovariateAnalysis@dev')

library(glmnet)
library(e1071)
library(randomForest)
library(kernlab)
library(nnet)
library(data.table)

synapseLogin()

args <- commandArgs(trailingOnly = TRUE)
featsynid <- args[1]  # assumes all individuals (training and test) and all features (walk and rest) in single tsv file at this synId
testset <- tolower(args[2]) # options are leaderboard, final or none



#trainingfeatureids<-c(rest="syn9952416", walk="syn9920447", walkmetadata="syn10138914", restmetadata="syn10142011")
trainingfeatureids<-c(features=featsynid, walkmetadata="syn10138914", restmetadata="syn10142011")
traininghcids<-"syn9819504"

demotrainquery<-"SELECT * FROM syn9819507"
demotrainqueryResult <- synTableQuery(demotrainquery)

trainingfeatures = lapply(trainingfeatureids, downloadFile)
trainingdemo<-demotrainqueryResult@values

bad.records.walk<-c("98565b66-1ba0-4f23-80a6-d4fedeb9a26c", "2d12ee94-522d-4cbc-b6cd-3dd1d8b35d27")
bad.records.demo<-c("cc8cc0f3-98b8-481d-aaa3-3b18028261fa")
bad.records.rest<-NULL

###########
# Walking test
############

# Prepare Data
walkvariables<-names(trainingfeatures$features)
walkvariables<-walkvariables[grep("walk", walkvariables)]
#walkvariables<-walkvariables[!tolower(walkvariables) %in% c("recordid","healthcode")]
trainingwalk<-merge.data.frame(trainingfeatures$walkmetadata, trainingfeatures$features[,c("healthCode", "recordId", walkvariables)], by=c("healthCode","recordId"), all.x=T, sort=F)
trainingwalk<-merge.data.frame(trainingdemo, trainingwalk, by=c("healthCode"), suffixes=c(".demo", ".walktest"), all.y=T, sort=F)
trainingwalk<-trainingwalk[trainingwalk$error=="",]   #Comment out for now

training_hcs_inconsistent<-unique(trainingwalk$healthCode[trainingwalk$`professional-diagnosis`==FALSE&trainingwalk$medTimepoint%in%c("Immediately before Parkinson medication", "Just after Parkinson medication (at your best)")])
training_hcs_inconsistent<-training_hcs_inconsistent[!is.na(training_hcs_inconsistent)]

trainingwalk<-trainingwalk[!trainingwalk$healthCode %in% training_hcs_inconsistent,]
trainingwalk<-trainingwalk[!trainingwalk$recordId.walktest %in% bad.records.walk & !trainingwalk$recordId.demo %in% bad.records.demo,]


# Summarized by Median

groupvariables<-c("healthCode", "medTimepoint", "professional-diagnosis", "age", "gender", "appVersion.walktest", "phoneInfo.walktest")

dttrainwalk<-data.table(trainingwalk)
dttrainwalk$count<-1

mediantrainingwalk<-dttrainwalk[, lapply(.SD, median, na.rm=TRUE), by=groupvariables, .SDcols=walkvariables ]
tmpcount<-dttrainwalk[, .(.N), by = groupvariables]
mediantrainingwalk<-merge.data.frame(mediantrainingwalk, tmpcount, by=groupvariables)

#
# Without App and Phone info
# Apply machine learning excluding medicated timepoints
#
trainsubsetwalk<-mediantrainingwalk[mediantrainingwalk$age>=57&mediantrainingwalk$medTimepoint!="Just after Parkinson medication (at your best)"&mediantrainingwalk$medTimepoint!=""&mediantrainingwalk$medTimepoint!="Another time",]
trainsubsetwalk<-trainsubsetwalk[!is.na(trainsubsetwalk$`professional-diagnosis`),]
trainoutcome<-trainsubsetwalk$`professional-diagnosis`
trainoutcome<-factor(ifelse(trainoutcome, "PD", "Control"))
trainoutcome<-factor(trainoutcome)

trainfeatureswalk_nophoneapp<-cbind(trainsubsetwalk[,c("age", "gender")], trainsubsetwalk[,walkvariables])
trainfeatureswalk_nophoneapp<-trainfeatureswalk_nophoneapp[,!names(trainfeatureswalk_nophoneapp)%in%c("steps_time_in_seconds", "numberOfSteps", "distance")]
trainfeatureswalk_nophoneapp[,"gender"]<-as.factor(trainfeatureswalk_nophoneapp[,"gender"])

set.seed(1234)
myControl_nophoneapp <- trainControl(  method="boot",
                                       number = 50,
                                       savePredictions="final",
                                       classProbs=TRUE,
                                       index=createResample(trainoutcome, 50),
                                       #                            seeds = seeds,
                                       summaryFunction=twoClassSummary )


tmpmmatrix_nophoneapp<-model.matrix( ~ ., trainfeatureswalk_nophoneapp)
tmpmmatrix_nophoneapp<-tmpmmatrix_nophoneapp[,-1]

model_list_nophoneapp <- caretList(
  y=trainoutcome , x=tmpmmatrix_nophoneapp,
  trControl=myControl_nophoneapp,
  metric = "ROC",
  tuneList=list(enet = caretModelSpec(method = "glmnet"), 
                rf1 = caretModelSpec(method="rf"), 
                svm = caretModelSpec(method="svmLinear"), 
                knn = caretModelSpec(method="knn"), 
                nnet = caretModelSpec(method = "nnet", trace=F)),
  continue_on_fail = FALSE, preProc = c("center", "scale")
)

greedy_ensemble_nophoneapp <- caretEnsemble(
  model_list_nophoneapp, 
  metric="ROC",
  trControl=trainControl(
    number=length(model_list_nophoneapp),
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))
#summary(greedy_ensemble_nophoneapp)


###########
# Standing test
############

# Prepare Data
restvariables<-names(trainingfeatures$features)
restvariables<-restvariables[grep("rest", restvariables)]
trainingrest<-merge.data.frame(trainingfeatures$restmetadata, trainingfeatures$features[,c("healthCode", "recordId", restvariables)], by=c("healthCode","recordId"), all.x=T, sort=F)
trainingrest<-merge.data.frame(trainingdemo, trainingrest, by=c("healthCode"), suffixes=c(".demo", ".resttest"), all.y=T, sort=F)
trainingrest<-trainingrest[trainingrest$error=="None",]   #Comment out for now

training_hcs_inconsistent_rest<-unique(trainingrest$healthCode[trainingrest$`professional-diagnosis`==FALSE&trainingrest$medTimepoint%in%c("Immediately before Parkinson medication", "Just after Parkinson medication (at your best)")])
training_hcs_inconsistent_rest<-training_hcs_inconsistent_rest[!is.na(training_hcs_inconsistent_rest)]

trainingrest<-trainingrest[!trainingrest$healthCode %in% training_hcs_inconsistent_rest,]
trainingrest<-trainingrest[!trainingrest$recordId.resttest %in% bad.records.rest & !trainingrest$recordId.demo %in% bad.records.demo,]


# Summarized by Median

groupvariables.rest<-c("healthCode", "medTimepoint", "professional-diagnosis", "age", "gender", "appVersion.resttest", "phoneInfo.resttest")

dttrainrest<-data.table(trainingrest)
dttrainrest$count<-1

mediantrainingrest<-dttrainrest[, lapply(.SD, median, na.rm=TRUE), by=groupvariables.rest, .SDcols=restvariables ]
tmpcount<-dttrainrest[, .(.N), by = groupvariables.rest]
mediantrainingrest<-merge.data.frame(mediantrainingrest, tmpcount, by=groupvariables.rest)


#
# Without App and Phone info
# Apply machine learning excluding medicated timepoints
#
trainsubsetrest<-mediantrainingrest[mediantrainingrest$age>=57&mediantrainingrest$medTimepoint!="Just after Parkinson medication (at your best)"&mediantrainingrest$medTimepoint!=""&mediantrainingrest$medTimepoint!="Another time",]
trainsubsetrest<-trainsubsetrest[!is.na(trainsubsetrest$`professional-diagnosis`),]
trainoutcome_rest<-trainsubsetrest$`professional-diagnosis`
trainoutcome_rest<-factor(ifelse(trainoutcome_rest, "PD", "Control"))
trainoutcome_rest<-factor(trainoutcome_rest)

trainfeaturesrest_nophoneapp<-cbind(trainsubsetrest[,c("age", "gender")], trainsubsetrest[,restvariables])
trainfeaturesrest_nophoneapp[,"gender"]<-as.factor(trainfeaturesrest_nophoneapp[,"gender"])

set.seed(5432)
myControl_rest_nophoneapp <- trainControl(  method="boot",
                                            number = 50,
                                            savePredictions="final",
                                            classProbs=TRUE,
                                            index=createResample(trainoutcome_rest, 50),
                                            #                            seeds = seeds,
                                            summaryFunction=twoClassSummary )


tmpmmatrix_rest_nophoneapp<-model.matrix( ~ ., trainfeaturesrest_nophoneapp)
tmpmmatrix_rest_nophoneapp<-tmpmmatrix_rest_nophoneapp[,-1]

model_list_rest_nophoneapp <- caretList(
  y=trainoutcome_rest , x=tmpmmatrix_rest_nophoneapp,
  trControl=myControl_rest_nophoneapp,
  metric = "ROC",
  tuneList=list(enet = caretModelSpec(method = "glmnet"), 
                rf1 = caretModelSpec(method="rf"), 
                svm = caretModelSpec(method="svmLinear"), 
                knn = caretModelSpec(method="knn"), 
                nnet = caretModelSpec(method = "nnet", trace=F)),
  continue_on_fail = FALSE, preProc = c("center", "scale")
)


greedy_ensemble_rest_nophoneapp <- caretEnsemble(
  model_list_rest_nophoneapp, 
  metric="ROC",
  trControl=trainControl(
    number=length(model_list_rest_nophoneapp),
    summaryFunction=twoClassSummary,
    classProbs=TRUE
  ))
summary(greedy_ensemble_rest_nophoneapp)


#############################
##
## Download test data and test
##
#############################

if(testset!="none"){
  if(testset=="leaderboard"){
    demorestrictedquery<-"SELECT * FROM syn9819508"
    demorestrictedqueryResult <- synTableQuery(demorestrictedquery)

    testfeatureids<-c(walkmetadata="syn10138365", restmetadata="syn10140543")
    testfeatures = lapply(testfeatureids, downloadFile)
    testdemo<-demorestrictedqueryResult@values
  } else if(testset=="final"){
    #broken for now
  }

  ###########
  # Walking test
  ############

  # Prepare Data
  testwalk<-merge.data.frame(testfeatures$walkmetadata, testfeatures$features[,c("healthCode", "recordId", walkvariables)], by=c("healthCode","recordId"), all.x=T, sort=F)
  testwalk<-merge.data.frame(testdemo, testwalk, by=c("healthCode"), suffixes=c(".demo", ".walktest"), all.y=T, sort=F)
  testwalk<-testwalk[testwalk$error=="",]
  test_hcs_inconsistent<-unique(testwalk$healthCode[testwalk$`professional-diagnosis`==FALSE&testwalk$medTimepoint%in%c("Immediately before Parkinson medication", "Just after Parkinson medication (at your best)")])
  test_hcs_inconsistent<-test_hcs_inconsistent[!is.na(test_hcs_inconsistent)]
  testwalk<-testwalk[!testwalk$healthCode %in% test_hcs_inconsistent,]

  # Summarized by Median

  dttestwalk<-data.table(testwalk)
  mediantestwalk<-dttestwalk[, lapply(.SD, median, na.rm=TRUE), by=groupvariables, .SDcols=walkvariables ]
  tmpcount<-dttestwalk[, .(.N), by = groupvariables]
  mediantestwalk<-merge.data.frame(mediantestwalk, tmpcount, by=groupvariables)

  testsubsetwalk<-mediantestwalk[mediantestwalk$age>=57&mediantestwalk$medTimepoint!="Just after Parkinson medication (at your best)"&mediantestwalk$medTimepoint!=""&mediantestwalk$medTimepoint!="Another time",]
  testsubsetwalk<-testsubsetwalk[!is.na(testsubsetwalk$`professional-diagnosis`),]
  testoutcome<-testsubsetwalk$`professional-diagnosis`
  testoutcome<-factor(ifelse(testoutcome, "PD", "Control"))
  testoutcome<-factor(testoutcome)

  testfeatureswalk_nophoneapp<-cbind(testsubsetwalk[,c("age", "gender")], testsubsetwalk[,walkvariables])
  testfeatureswalk_nophoneapp<-testfeatureswalk_nophoneapp[,!names(testfeatureswalk_nophoneapp)%in%c("steps_time_in_seconds", "numberOfSteps", "distance")]
  testfeatureswalk_nophoneapp[,"gender"]<-as.factor(testfeatureswalk_nophoneapp[,"gender"])

  tmptestmmatrix_nophoneapp<-model.matrix( ~ ., testfeatureswalk_nophoneapp)
  tmptestmmatrix_nophoneapp<-tmptestmmatrix_nophoneapp[,-1]



  ###########
  # Standing test
  ############

  # Prepare Data
  testrest<-merge.data.frame(testfeatures$restmetadata, testfeatures$features[,c("healthCode", "recordId", restvariables)], by=c("healthCode","recordId"), all.x=T, sort=F)
  testrest<-merge.data.frame(testdemo, testrest, by=c("healthCode"), suffixes=c(".demo", ".resttest"), all.y=T, sort=F)
  testrest<-testrest[testrest$error=="None",]
  test_hcs_inconsistent_rest<-unique(testrest$healthCode[testrest$`professional-diagnosis`==FALSE&testrest$medTimepoint%in%c("Immediately before Parkinson medication", "Just after Parkinson medication (at your best)")])
  test_hcs_inconsistent_rest<-test_hcs_inconsistent_rest[!is.na(test_hcs_inconsistent_rest)]
  testrest<-testrest[!testrest$healthCode %in% test_hcs_inconsistent_rest,]

  # Summarized by Median
  dttestrest<-data.table(testrest)
  mediantestrest<-dttestrest[, lapply(.SD, median, na.rm=TRUE), by=groupvariables.rest, .SDcols=restvariables ]
  tmpcount<-dttestrest[, .(.N), by = groupvariables.rest]
  mediantestrest<-merge.data.frame(mediantestrest, tmpcount, by=groupvariables.rest)


  #
  # Without App and Phone info
  # Apply machine learning excluding medicated timepoints
  #
  testsubsetrest<-mediantestrest[mediantestrest$age>=57&mediantestrest$medTimepoint!="Just after Parkinson medication (at your best)"&mediantestrest$medTimepoint!=""&mediantestrest$medTimepoint!="Another time",]
  testsubsetrest<-testsubsetrest[!is.na(testsubsetrest$`professional-diagnosis`),]
  testoutcome_rest<-testsubsetrest$`professional-diagnosis`
  testoutcome_rest<-factor(ifelse(testoutcome_rest, "PD", "Control"))
  testoutcome_rest<-factor(testoutcome_rest)

  testfeaturesrest_nophoneapp<-cbind(testsubsetrest[,c("age", "gender")], testsubsetrest[,restvariables])
  testfeaturesrest_nophoneapp[,"gender"]<-as.factor(testfeaturesrest_nophoneapp[,"gender"])

  tmptestmmatrix_rest_nophoneapp<-model.matrix( ~ ., testfeaturesrest_nophoneapp)
  tmptestmmatrix_rest_nophoneapp<-tmptestmmatrix_rest_nophoneapp[,-1]

} else {
  testsubsetwalk<-trainsubsetwalk
  testsubsetrest<-trainsubsetrest
  testoutcome<-trainoutcome
  testoutcome_rest<-testoutcome
  tmptestmmatrix_nophoneapp <-tmpmmatrix_nophoneapp
  tmptestmmatrix_rest_nophoneapp <- tmpmmatrix_rest_nophoneapp
}
  
preds_allmodels_nophoneapp <- as.data.frame(predict(object=model_list_nophoneapp, tmptestmmatrix_nophoneapp))
preds_greedensemble_nophoneapp <- predict(object=greedy_ensemble_nophoneapp, tmptestmmatrix_nophoneapp, type="prob")
preds_allmodels_nophoneapp$ensemble<-preds_greedensemble_nophoneapp
auc_allmodels_nophoneapp <- apply(preds_allmodels_nophoneapp, 2, function(x, testoutcome) {
  return(roc(ifelse(testoutcome=="PD",1,0), x)$auc)
}, testoutcome=testoutcome)
#  print(auc_allmodels_nophoneapp)



preds_allmodels_rest_nophoneapp <- as.data.frame(predict(object=model_list_rest_nophoneapp, tmptestmmatrix_rest_nophoneapp))
preds_greedensemble_rest_nophoneapp <- predict(object=greedy_ensemble_rest_nophoneapp, tmptestmmatrix_rest_nophoneapp, type="prob")
preds_allmodels_rest_nophoneapp$ensemble<-preds_greedensemble_rest_nophoneapp
auc_allmodels_rest_nophoneapp <- apply(preds_allmodels_rest_nophoneapp, 2, function(x, testoutcome) {
  return(roc(ifelse(testoutcome=="PD",1,0), x)$auc)
}, testoutcome=testoutcome_rest)
#  print(auc_allmodels_rest_nophoneapp) 

out_walk<-data.frame(testsubsetwalk[,c("healthCode", "medTimepoint")], walk.predictions=preds_allmodels_nophoneapp$ensemble, stringsAsFactors = FALSE)
out_rest<-data.frame(testsubsetrest[,c("healthCode", "medTimepoint")], rest.predictions=preds_allmodels_rest_nophoneapp$ensemble, stringsAsFactors = FALSE)

write.table(merge.data.frame(out_walk, out_rest, by=c("healthCode", "medTimepoint"), all=T, sort = FALSE), paste(featsynid, "_", testset, ".txt"), row.names=F, quote=F)

scores<-c(walk=auc_allmodels_nophoneapp$ensemble, rest=auc_allmodels_rest_nophoneapp$ensemble)
cat(scores)
