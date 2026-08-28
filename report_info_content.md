# Trajectory Information Content - Analysis Report

## 1. Transient vs Stationary Regimes
Overall, how do the information metrics differ between the initial transient relaxation and the later stationary fluctuations?

```text
                         TDMI         eSNR       TIC       Var
Regime     Topology                                           
Stationary BA        0.636046    21.110351  0.047370  0.084920
           ER        0.658052   100.183809  0.076832  0.109125
           SBM       0.653174    43.176147  0.070656  0.143891
Transient  BA        1.371088  3867.283477  1.324578  2.916990
           ER        1.339215  4592.620521  1.408050  3.786326
           SBM       1.331596  5450.292435  1.422443  3.771165
```
*Observation*: As expected, the variance and the deterministic signal-to-noise ratio (eSNR) are generally higher in the Transient regime compared to the Stationary regime, where the system mostly fluctuates around an attractor.

## 2. Impact of Measurement Interval ($dt_{meas}$)
```text
                        TDMI          eSNR       TIC
Regime     dt_meas                                  
Stationary 0.05     1.426147    194.277308  0.112136
           0.10     1.104833     63.609995  0.101762
           0.50     0.419723     11.921429  0.054191
           1.00     0.208989      3.521646  0.042918
           2.00     0.085762      0.786801  0.013756
Transient  0.05     2.111280  12002.187416  1.811661
           0.10     1.744781   5963.709451  1.637926
           0.50     0.857393    504.384855  0.991196
           1.00     0.714796    114.970284  1.066865
           2.00     0.597642      0.000000  1.164201
```
*Observation*: As $dt_{meas}$ increases, the Time-Delayed Mutual Information (TDMI) decreases because the consecutive points become more decorrelated.

## 3. Node-Wise Topological Correlations
Does the degree of a node predict how much information its trajectory contains?

### Transient Regime
- **In-Degree** correlation with TIC: 0.016, TDMI: -0.036, eSNR: 0.019

### Stationary Regime
- **In-Degree** correlation with TIC: 0.048, TDMI: 0.002, eSNR: 0.015
