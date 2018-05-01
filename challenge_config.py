# Use rpy2 if you have R scoring functions
import rpy2.robjects as robjects
import pandas as pd
import os
from rpy2.robjects import pandas2ri
from csv import Sniffer
pandas2ri.activate()
filePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),'PD_Challenge_SC1_Scoring_Functions.R')
# filePath = os.path.join(os.path.dirname(os.path.abspath('PD_Challenge_SC1_Scoring_Functions.R')), 'PD_Challenge_SC1_Scoring_Functions.R')
robjects.r("source('%s')" % filePath)
PD_score_challenge = robjects.r('PD_score_challenge')


########### login to synapse
########### This part needs to be changed to read from the syanpseConfig file on the machine
import synapseclient
syn = synapseclient.login()


##-----------------------------------------------------------------------------
##
## challenge specific code and configuration
##
##-----------------------------------------------------------------------------

## A Synapse project will hold the assets for your challenge. Put its
## synapse ID here, for example
## CHALLENGE_SYN_ID = "syn1234567"
CHALLENGE_SYN_ID = "syn8717496"

## Name of your challenge, defaults to the name of the challenge's project
CHALLENGE_NAME = "Parkinsons Disease Digital Biomarker DREAM Challenge"

## Synapse user IDs of the challenge admins who will be notified by email
## about errors in the scoring script
ADMIN_USER_IDS = [123456]
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
    # CHECK: that the file is .csv format
    assert submission.entity.contentType == 'text/csv', "Submission file must be in .csv format"

    # CHECK: that the file isn't empty
    assert os.stat(submission.filePath).st_size > 0, "Prediction file can't be empty"
    # headers that are required (only recordId)
    REQUIRED_HEADERS = ["recordId"]
    # read in submission file
    pred = pd.read_csv(submission.filePath)
    # CHECK: All required headers exist
    assert all([i in pred.columns for i in REQUIRED_HEADERS]), "Submission file must have header: recordId"
    # CHECK: No duplicate samples allowed
    assert sum(pred['recordId'].duplicated()) == 0, "No duplicated submissions allowed"
    # CHECK: No NA's are allowed
    assert pred.isnull().values.sum() == 0, "There cannot be any empty values"
    # CHECK: all values other than recordId must be numeric
    pred1 = pred.ix[:, pred.columns != 'recordId'] # get all columns except recordId
    foo = pred1.applymap(lambda x: isinstance(x, float)) # convert all values to True/False depending on if they're a float or not
    assert all(foo.apply(lambda x: all(x))), "All values except recordId must be numeric" # fail if there is a False

    # CHECK: all recordIds are present (syn10626200)
    test_submission =  os.getcwd() + '/recordID_File.csv' # file with all the recordIds
    record_ids = set(pd.read_csv(test_submission, header=0).recordId) # get just the recordID column
    user_record_ids = set(pred.recordId)
    for i in record_ids:
        assert i in user_record_ids, "Not all recordIds are present in column: recordId"

    # get list of everyone that has access to data (AR 9602969)
    accessIdList = []
    for i in range(0, 500, 50):
        bar = syn.restGET('https://repo-prod.prod.sagebase.org/repo/v1/entity/syn10146553/accessApproval?limit=50&offset=%s' %i) # REST call of everyone with access
        accessIds = [b['accessorId'] for b in bar['results']] # get userId
        accessIdList.append(accessIds)  # append ids to list

    accessList = [item for nestedList in accessIdList for item in nestedList] # un-nest the nested list        

    # All members of a team must have access to data (AR 9602969)
    listy = syn.getSubmission(submission.id).contributors # get info on the contributors of the team
    members = [x['principalId'] for x in listy] # this gets the ids of the members of the team

    temp = [item for item in members if item not in accessList] # if the team member is not part of the access list, store them to a list
    assert temp ==[], ("Not all members have access to the data. Please verify that " + str(temp) + " have gone through the full validation process.")

    return(True,"Passed Validation")

def score_subchallenge_one(submission, goldstandard):
    import pandas as pd
    import synapseclient
    syn = synapseclient.login()
    # read in submission
    # pred = pd.read_csv(submission.filePath)
    pred = submission.filePath

    # get submission synId for provenance
    submissionId = submission.entityId

    # score the submission
    score_challenge = PD_score_challenge(pred, walk=True, leaderboard=True, permute=False, nperm=10000, filename='submission.csv', parentSynId="syn10347832", submissionSynId=submissionId)

    # Return the score
    return(score_challenge)

def score2(submission, goldstandard_path):
    ##Read in submission (submission.filePath)
    ##Score against goldstandard
    return(score1, score2, score3)

evaluation_queues = [
    {
        'id': 9606375, # mpower
        'scoring_func':score_subchallenge_one,
        'validation_func':validate_func,
        'goldstandard_path':'path/to/sc1gold.txt'
    }
    # ,
    # {
    #     'id': 9606376, # actionTremor
    #     'scoring_func':score2,
    #     'validation_func':validate_func,
    #     'goldstandard_path':'path/to/sc2gold.txt'

    # }
    # ,
    # {
    #     'id': 9606377, # dyskinesia
    #     'scoring_func':score3,
    #     'validation_func':validate_func,
    #     'goldstandard_path':'path/to/sc2gold.txt'

    # }
    # ,
    # {
    #     'id': 9606378, # bradykinesia
    #     'scoring_func':score4,
    #     'validation_func':validate_func,
    #     'goldstandard_path':'path/to/sc2gold.txt'

    # }
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
    auc, pval = config['scoring_func'](submission, config['goldstandard_path'])
    #Make sure to round results to 3 or 4 digits
    return (dict(auroc=auc[0], pval=pval[0]), "You did fine!")
