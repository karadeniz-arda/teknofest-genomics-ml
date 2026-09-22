import json
import numpy as np
import pandas as pd
import pytest

from teknofest_genomics.models.inference import predict

class DummyModel:
    classes_ = np.array([0, 1])
    def __init__(self, probability: float = .7): self.probability = probability
    def predict_proba(self, X): return np.column_stack([np.full(len(X), 1-self.probability), np.full(len(X), self.probability)])

def artifact(probability=.7):
    model=DummyModel(probability)
    return {'xgb_model':model,'lgb_model':model,'cat_model':model,'weights':{'xgb':.1,'lgb':.1,'cat':.8},'threshold':.5,'features':['category','number'],'categorical_features':['category'],'categories':{'category':['known','_MISSING_','_UNK_']}}

def frame(): return pd.DataFrame({'Variant_ID':['HIDDEN_A','HIDDEN_B'],'category':['unseen',None],'number':[1.,2.]})

def test_id_category_probability_and_json_safety():
    result=predict(frame(),artifact(.7))
    assert result.id.tolist()==['HIDDEN_A','HIDDEN_B']
    assert result.predicted_class.tolist()==['1','1']
    assert np.isfinite(result.predicted_prob).all()
    assert result.predicted_prob.between(0,1).all()
    json.dumps(result.to_dict('records'),allow_nan=False)

def test_threshold_maps_to_zero_and_one_strings():
    assert predict(frame(),artifact(.2)).predicted_class.tolist()==['0','0']
    assert predict(frame(),artifact(.8)).predicted_class.tolist()==['1','1']

def test_duplicate_ids_are_rejected():
    bad=frame(); bad.loc[1,'Variant_ID']='HIDDEN_A'
    with pytest.raises(ValueError,match='unique'):
        predict(bad,artifact())

def test_out_of_range_probability_is_rejected():
    with pytest.raises(ValueError,match='out-of-range'):
        predict(frame(),artifact(1.2))
