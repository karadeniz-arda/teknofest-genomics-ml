import numpy as np
from teknofest_genomics.evaluation.metrics import evaluate_binary

def test_f1_mcc_and_official_score():
    y=np.array([0,0,1,1]); probability=np.array([.1,.9,.8,.2])
    result=evaluate_binary(y,probability,.5)
    assert result['tn']==1 and result['fp']==1 and result['fn']==1 and result['tp']==1
    assert result['f1']==.5
    assert result['mcc']==0.0
    assert result['official_score']==.25
