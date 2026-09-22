import pandas as pd
from teknofest_genomics.data.preprocessing import apply_schema, fit_schema
def test_ids_and_unseen_categories_are_safe():
 train=pd.DataFrame({'Variant_ID':['a','b'],'Label':[0,1],'feature':['x',None],'number':[1.,2.]})
 schema=fit_schema(train); test=pd.DataFrame({'Variant_ID':['HIDDEN_A','HIDDEN_B'],'feature':['new',None],'number':[3.,4.]})
 result=apply_schema(test,schema)
 assert result.Variant_ID.tolist()==['HIDDEN_A','HIDDEN_B']
 assert result.feature.astype(str).tolist()==['_UNK_','_MISSING_']
