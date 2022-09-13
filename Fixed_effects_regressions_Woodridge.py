# Example 14.2 in Woodridge "Introductory Econometrics: A Modern Approach"
# within transformation approach 

import wooldridge as woo
import pandas as pd
import linearmodels as plm

# changes in return to education 
wagepan = woo.dataWoo('wagepan')

# index variables 'nr', 'year': for individuals and years
wagepan = wagepan.set_index(['nr', 'year'], drop=False)

# FE model estimation:
reg = plm.PanelOLS.from_formula(
    formula='lwage ~ married + union + C(year)*educ + EntityEffects',
    data=wagepan, drop_absorbed=True)
results = reg.fit()

# print regression table:
table = pd.DataFrame({'b': round(results.params, 4),
                      'se': round(results.std_errors, 4),
                      't': round(results.tstats, 4),
                      'pval': round(results.pvalues, 4)})
print(f'table: \n{table}\n')
