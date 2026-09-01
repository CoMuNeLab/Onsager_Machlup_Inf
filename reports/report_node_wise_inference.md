# Node-Wise Trajectory Information and Network Inference Report

This report evaluates whether the possibility of reconstructing the incoming parameters associated with a *single node* is predicted by the information content of its *own trajectory*.

## 1. Spearman Rank Correlations

Correlation between node-specific trajectory metrics and inference performance metrics for that node's incoming edges:

           var_x  mean_drift_sq  esnr_vel      tdmi  in_degree
Target                                                        
AUROC   0.227376       0.239688  0.288047  0.200772  -0.089317
AUPRC   0.239234       0.229240  0.333979  0.223884   0.344025
MSE     0.090292       0.077911  0.107681  0.069026   0.843544
F1      0.237057       0.236810  0.351451  0.223427  -0.088445
SHD     0.114785       0.103446  0.136883  0.092662   0.811999

## 2. Standardized OLS Regression (AUROC)

Model: `AUROC ~ log_M + var_x + mean_drift_sq + esnr_vel` (all standardized)

```text
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  AUROC   R-squared:                       0.132
Model:                            OLS   Adj. R-squared:                  0.126
Method:                 Least Squares   F-statistic:                     20.81
Date:                lun, 17 ago 2026   Prob (F-statistic):           5.57e-16
Time:                        11:17:12   Log-Likelihood:                -743.66
No. Observations:                 552   AIC:                             1497.
Df Residuals:                     547   BIC:                             1519.
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
=================================================================================
                    coef    std err          t      P>|t|      [0.025      0.975]
---------------------------------------------------------------------------------
Intercept        -1.7911      0.271     -6.616      0.000      -2.323      -1.259
log_M             0.7623      0.114      6.689      0.000       0.538       0.986
var_x             0.2409      0.146      1.654      0.099      -0.045       0.527
mean_drift_sq    -0.3300      0.156     -2.112      0.035      -0.637      -0.023
esnr_vel          0.3037      0.056      5.433      0.000       0.194       0.413
==============================================================================
Omnibus:                       34.623   Durbin-Watson:                   2.022
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               39.437
Skew:                           0.626   Prob(JB):                     2.73e-09
Kurtosis:                       3.382   Cond. No.                         19.0
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

## 3. ANOVA Variance Decomposition

Which factor explains the most variance in node-wise AUROC?

| Factor | Sum of Squares | F | p-value | % Variance |
| :--- | :---: | :---: | :---: | :---: |
| `log_M` | 39.1196 | 44.74 | 5.56e-11 | **7.12%** |
| `var_x` | 2.3926 | 2.74 | 9.86e-02 | **0.44%** |
| `mean_drift_sq` | 3.8993 | 4.46 | 3.52e-02 | **0.71%** |
| `esnr_vel` | 25.8096 | 29.52 | 8.34e-08 | **4.70%** |
| `Residual` | 478.2318 | nan | nan | **87.04%** |

## 4. Conclusion

The analysis clearly shows whether individual node trajectory content, independent of the rest of the network, determines parameter reconstructability.
