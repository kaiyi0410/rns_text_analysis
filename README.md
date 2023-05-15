# rns_text_analysis

## Code files
*1_multi_horizon_returns.py*
  - creates a random subset of 10,000 RNS statements from 2010-2022
  - calculates 20-day returns for each RNS in the subset

*2_sentiment_analysis.py*
  - TF-IDF vectorising and calculating sentiment metrics 
  - creates word clouds for the 50 most frequently-appearing companies 

*3_similarity_network_GLM.R*
  - calculates readability metrics
  - calculates RNS similarity and plots network of 50 most frequently-appearing companies 
  - sets up GLMs to test associations between RNS metrics and returns 


## CSV files
*rns_final_metrics.csv*
Combined dataset of the metrics we calculated
Abbreviations:
  - ret20: 20-day returns
  - MSL: mean sentence length
  - MWL: mean word length
  - GFI: Gunning-Fog index

*ind50.csv*
Subset of 50 most frequently-appearing companies in our subset with their economic sector label 
