## [Vodafone music challenge](https://www.kaggle.com/c/better-vodafone-music-challenge)


##### Private(v2): 4th place 0.83069 (Best submission is 3th place)

##### Public(v2): 16th place 0.84456


#### Code
* V1 - simple preprocessing and fit predict
* V2 - level 2 and 3 categorical features interactions
* V3 - numerical features operation interactions (plus, minus, div, product)
* V4 - statistics features over different categories
* V5 - target encoding over double K-fold split (reference: BNP Paribas competition)

#### Validation

5 fold stratified validation with respect to class weights because training data is very imbalanced (10/90 target distribution).
Also, I believe undersampling is a solution too. 

