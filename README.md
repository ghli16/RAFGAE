# RAFGAE

Code and Dataset for "Drug repositioning based on residul attention network and free multiscale adversarial training".

## Installation

Before starting your analysis, please have a look at the [RAFGAE Github](https://github.com/ghli16/RAFGAE/tree/main), you can download the full project code and open it in Pycharm's File Options New Project.

The code has been tested running under Windows and Python 3.6.8. The required packages are as follows:

- numpy == 1.19.5

- scipy == 1.15.4

- torch == 1.10.2

- torch-geometric == 2.0.3

- torch-scatter == 2.0.9

- torchvision == 0.11.3

## INPUT

### Benchmark Dataset

We employed two benchmark datasets established by investigators. The first is the F dataset, which corresponds to Gottlieb's gold standard dataset. It contains 1933 known associations between diseases and drugs, including 313 diseases collected from the OMIM database and 593 drugs obtained from the DrugBank database. The second is the C-dataset , which includes 2532 known associations between 409 diseases collected from the OMIM database and 663 drugs obtained from the DrugBank database.

* ``Dataset1`` \
 F dataset
* ``Dataset2`` \
 C dataset
* ``ddir.csv`` \
the drug_disease association matrix
* ``dis_ss.csv.csv`` \
 the disease semantic similarity matrix
* ``drug_ss.csv`` \
   the drug chemical_structure similarity matrix

### CODE

* ``main.py`` \
    To predict drug-disease associations by RAFGAE, run
* ``model.py`` \
     the GAT_ReN framework and GAE module;
* ``attack.py`` \
     the adversarial training module;
* ``train.py`` \
     the 10-fold cross validation


## OUTPUT
* ``results.csv`` \
 the prediction results

* ``Final result`` \
 the evaluation indicators
 
    MEAN_AUC: 0.9343 | MEAN_AUPR: 0.5270
    
    To evaluate the accuracy of RAFGAE, the receiver operat
ing characteristic (ROC) curve is used. It is plotted by two
variables including false positive rate and true positive
rate. Considering the biased performance of arear under
the curve (AUC) for imbalanced datasets, we also make
use of the precision–recall (PR) curve to precisely ref lect
the actual performance of prediction models. AUC and
AUPR are the areas under ROC and PR curves respectively,
and they are used to quantitatively indicate the perfor
mance in terms of AUC and PR. 


## Run
To run RAFGAE with default settings:
   * ``run main.py`` 

To assess the effectiveness of RAFGAE in predicting the known associations, 10-fold cross validation (CV) is applied, which uses nine-fold associations as the training set and utilizes the remaining one-fold to validate the performance of RAFGAE. This process is repeated 10 times, taking each fold alternately as testing data.
    
    Optional Argument:
      -fold Number of k-folds cross-validation
      -epochs Number of epoches
      -lr of Learning rate
      -weight_decay of Weight decay(L2 loss on parameters)
      -hidden Dimension of representations
      ...
It takes about 30 minutes to run the project code in its entirety

##MODEL
 
![RAFGAE architecture](https://github.com/ghli16/RAFGAE/blob/main/model_structure.png)

![RAFGAE computational_workflow](https://github.com/ghli16/RAFGAE/blob/main/computational_workflow.png)

| RAFGAE Algorithm                                                                            |
|---------------------------------------------------------------------------------------------|
| Input: drug similarity Xdr; disease similarity Xdi; initial interaction matrix A;           |
| Output: score matrix F;                                                                     |
| 1:   Construct drug-drug similarity network Xdr and disease-disease similarity network Xdi; |
| 2:   Construct the bipartite network defined by bipartite network G                         |
| 3:  Construct the initial embedding H(0)  do forward propagation by GAT_ReN Framework;      |
| 4:   for epoch = 1 → N do                                                                   |
| 5:         δ0 ← U(−e, e)                                                                    |
| 6:         for i = 1 → L do                                                                 |
| 7:              H(0) ←H(0) + δ0                                                             |
| 8:              H(l) ← GATsConv(H(l-1), A)                                                  |
| 9:              H(l) ← H(l) + α * H(0) + β * H(l−1);                                        |
| 10:         end for                                                                         |
| 11:       Compute Adr and Adi  respectively;                                                |
| 12:       Calculate Sdr and Sdi  respectively;                                              |
| 13:       Update Zdr and Zdi by GAE Encoder                                                 |
| 14:       Update Fdr and Fdi by GAE Decoder                                                 |
| 15:       Compute score value: F = αFdr + (1 − α)Fdi                                        |
| 16:       L(F, A).backward()                                                                |
| 17:       δt ← δt-1 + e· sign (grad (δt)) by FMAT module                                    |
| 18:  end for                                                                                |
| 19: Return F;                                                                               |

 

# Contact
We welcome you to contact us (email: ghli16@hnu.edu.cn) for any questions and cooperations.
