# Use rpy2 if you have R scoring functions
import rpy2.robjects as robjects
import pandas as pd
import os
filePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'PD_Challenge_SC1_Scoring_Functions.R') 
# filePath = os.path.join(os.path.dirname(os.path.abspath('PD_Challenge_SC1_Scoring_Functions.R')), 'PD_Challenge_SC1_Scoring_Functions.R')
robjects.r("source('%s')" % filePath) 
PD_score_challenge = robjects.r('PD_score_challenge')
download_merge_covariate = robjects.r('download_merge_covariate')
fit_model = robjects.r('fit_model')
score_model = robjects.r('score_model')

##-----------------------------------------------------------------------------
##
## challenge specific code and configuration
##
##-----------------------------------------------------------------------------

###############
# !!!!!!!!!!!!!
# THIS IS A TEST RUN FOR A FAKE CHALLENGE
# !!!!!!!!!!!!!
###############

## A Synapse project will hold the assetts for your challenge. Put its
## synapse ID here, for example
## CHALLENGE_SYN_ID = "syn1234567"
CHALLENGE_SYN_ID = "syn10139090"

## Name of your challenge, defaults to the name of the challenge's project
CHALLENGE_NAME = "yooreeExampleChallenge"

## Synapse user IDs of the challenge admins who will be notified by email
## about errors in the scoring script
ADMIN_USER_IDS = [3337572]
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


## Each question in your challenge should have an evaluation queue through
## which participants can submit their predictions or models. The queues
## should specify the challenge project as their content source. Queues
## can be created like so:
##   evaluation = syn.store(Evaluation(
##     name="My Challenge Q1",
##     description="Predict all the things!",
##     contentSource="syn1234567"))
## ...and found like this:
##   evaluations = list(syn.getEvaluationByContentSource('syn3375314'))
## Configuring them here as a list will save a round-trip to the server
## every time the script starts and you can link the challenge queues to
## the correct scoring/validation functions.  Predictions will be validated and 

def validate_func(submission, goldstandard_path):
    ## Read in submission (submission.filePath)
    ## Validate submission
    ## MUST USE ASSERTION ERRORS!!! 
    ## Only assertion errors will be returned to participants, all other errors will be returned to the admin
    # CHECK: that the file is named "predictions.tsv"
    assert os.path.basename(submission.filePath) == "prediction.csv", "Submission file must be named prediction.csv"
    # CHECK: that the files isn't empty
    assert os.stat(submission).st_size > 0, "Prediction file can't be empty"
    # headers that are required (only recordId)
    REQUIRED_HEADERS = ["recordId"]
    # read in gold standard file
    # gold = pd.read_csv(goldstandard_path)
    # read in submission file
    pred = pd.read_csv(submission)
    # CHECK: All required headers exist
    assert all([i in pred.columns for i in REQUIRED_HEADERS]), "predictions.tsv must have header: recordId"
    # CHECK: No duplicate samples allowed
    assert sum(pred['recordId'].duplicated()) == 0, "No duplicated submissions allowed"
    # CHECK: No NA's are allowed
    assert pred.isnull().values.sum() == 0, "There cannot be any empty values"
    # CHECK: all values other than recordId must be numeric
    pred1 = pred.ix[:, pred.columns != 'recordId'] # get all columns except recordId
    foo = pred1.applymap(lambda x: isinstance(x, float)) # convert all values to True/False depending on if they're a float or not
    assert all(foo.apply(lambda x: all(x))), "All values except recordId must be numeric" # fail if there is a False 
    # CHECK: column names except recordId are all appended with _walk or _rest
    assert all(foo.columns.str.contains('_walk') | foo.columns.str.contains('_rest')), "All columns except recordId must be appended with _walk or _rest"
    # CHECK: all recordIds are present
    #assert len(pred.recordId) == 46903, "Cannot be missing recordIds"

    return(True,"Passed Validation")

def score1(submission, goldstandard):
    # read in submission 
    pred = pd.read_csv(submission.filePath, sep='\t')
    # read in gold standard
    # gold = pd.read_csv(goldstandard)

    # make dataframes from submisson into walk_test, walk_train, rest_test, rest_train
    df = pd.DataFrame(pd.read_csv((syn.get('')).path)) # get entity of all recordIds and convert to dataframe
    train = ''
    test = ''


    score_challenge = PD_score_challenge(train, test, walk=TRUE, leaderboard=TRUE)

    ##Score against goldstandard
    return(score_challenge)

def score2(submission, goldstandard_path):
    ##Read in submission (submission.filePath)
    ##Score against goldstandard
    return(score1, score2, score3)

evaluation_queues = [
    {
        'id':1,
        'scoring_func':score1
        'validation_func':validate_func
        'goldstandard_path':'path/to/sc1gold.txt'
    },
    {
        'id':2,
        'scoring_func':score2
        'validation_func':validate_func
        'goldstandard_path':'path/to/sc2gold.txt'

    }
]
evaluation_queue_by_id = {q['id']:q for q in evaluation_queues}


## define the default set of columns that will make up the leaderboard
LEADERBOARD_COLUMNS = [
    dict(name='objectId',      display_name='ID',      columnType='STRING', maximumSize=20),
    dict(name='userId',        display_name='User',    columnType='STRING', maximumSize=20, renderer='userid'),
    dict(name='entityId',      display_name='Entity',  columnType='STRING', maximumSize=20, renderer='synapseid'),
    dict(name='versionNumber', display_name='Version', columnType='INTEGER'),
    dict(name='name',          display_name='Name',    columnType='STRING', maximumSize=240),
    dict(name='team',          display_name='Team',    columnType='STRING', maximumSize=240)]

## Here we're adding columns for the output of our scoring functions, score,
## rmse and auc to the basic leaderboard information. In general, different
## questions would typically have different scoring metrics.
leaderboard_columns = {}
for q in evaluation_queues:
    leaderboard_columns[q['id']] = LEADERBOARD_COLUMNS + [
        dict(name='score',         display_name='Score',   columnType='DOUBLE'),
        dict(name='rmse',          display_name='RMSE',    columnType='DOUBLE'),
        dict(name='auc',           display_name='AUC',     columnType='DOUBLE')]

## map each evaluation queues to the synapse ID of a table object
## where the table holds a leaderboard for that question
leaderboard_tables = {}


def validate_submission(evaluation, submission):
    """
    Find the right validation function and validate the submission.

    :returns: (True, message) if validated, (False, message) if
              validation fails or throws exception
    """
    config = evaluation_queue_by_id[int(evaluation.id)]
    validated, validation_message = config['validation_func'](submission, config['goldstandard_path'])

    return True, validation_message


def score_submission(evaluation, submission):
    """
    Find the right scoring function and score the submission

    :returns: (score, message) where score is a dict of stats and message
              is text for display to user
    """
    config = evaluation_queue_by_id[int(evaluation.id)]
    score = config['scoring_func'](submission, config['goldstandard_path'])
    #Make sure to round results to 3 or 4 digits
    return (dict(score=round(score[0],4), rmse=score[1], auc=score[2]), "You did fine!")


