Drug repositioning based on residual attention network and free multiscale adversarial training


## Code
### Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
- numpy == 1.19.5
- scipy == 1.15.4
- pytorch == 1.10.2

Files: 
1.dataset
 1. ddir.csv stores known drug-disease association information; 
 2. drug_ss stores drug similarity matrix
 3. dis_ss stores disease similarity matrix;


 
 
2.code      

1.model.py：the GAT_ReN framework and GAE module；  
1.attack.py: the adversarial training module;     
 

