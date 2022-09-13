# Example 13.2 in Woodridge "Introductory Econometrics: A Modern Approach"

import wooldridge as woo
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

#effect of a garbage incerrator's location on housing prices
kielmc = woo.dataWoo('kielmc')

# separate regressions for 1978 and 1981:
y78 = (kielmc['year'] == 1978)
reg78 = smf.ols(formula='rprice ~ nearinc', data=kielmc, subset=y78)
results78 = reg78.fit()

y81 = (kielmc['year'] == 1981)
reg81 = smf.ols(formula='rprice ~ nearinc', data=kielmc, subset=y81)
results81 = reg81.fit()

# joint regression including an interaction term:
reg_joint = smf.ols(formula='rprice ~ nearinc * C(year)', data=kielmc)
results_joint = reg_joint.fit()

# print regression tables:
table_78 = pd.DataFrame({'b': round(results78.params, 4),
                         'se': round(results78.bse, 4),
                         't': round(results78.tvalues, 4),
                         'pval': round(results78.pvalues, 4)})
print(f'table_78: \n{table_78}\n')

table_81 = pd.DataFrame({'b': round(results81.params, 4),
                         'se': round(results81.bse, 4),
                         't': round(results81.tvalues, 4),
                         'pval': round(results81.pvalues, 4)})
print(f'table_81: \n{table_81}\n')

table_joint = pd.DataFrame({'b': round(results_joint.params, 4),
                            'se': round(results_joint.bse, 4),
                            't': round(results_joint.tvalues, 4),
                            'pval': round(results_joint.pvalues, 4)})
print(f'table_joint: \n{table_joint}\n')

# difference in difference (DiD):
reg_did = smf.ols(formula='np.log(rprice) ~ nearinc*C(year)', data=kielmc)
results_did = reg_did.fit()

# print regression table:
table_did = pd.DataFrame({'b': round(results_did.params, 4),
                          'se': round(results_did.bse, 4),
                          't': round(results_did.tvalues, 4),
                          'pval': round(results_did.pvalues, 4)})
print(f'table_did: \n{table_did}\n')

# DiD with control variables:
reg_didC = smf.ols(formula='np.log(rprice) ~ nearinc*C(year) + age +'
                           'I(age**2) + np.log(intst) + np.log(land) +'
                           'np.log(area) + rooms + baths',
                   data=kielmc)
results_didC = reg_didC.fit()

# print regression table:
table_didC = pd.DataFrame({'b': round(results_didC.params, 4),
                           'se': round(results_didC.bse, 4),
                           't': round(results_didC.tvalues, 4),
                           'pval': round(results_didC.pvalues, 4)})
print(f'table_didC: \n{table_didC}\n')

