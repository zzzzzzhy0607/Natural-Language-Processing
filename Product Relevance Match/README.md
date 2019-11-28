# ANLY580 Project
## Home Depot Product Search Relevance
## 400 Bad Request -- Hongyang Zheng, Heng Zhou, Zhengqian Xu
 
### Segregation of Duties
`Hongyang Zheng` is mainly responsible for data collection, data cleaning, and feature engineering parts.

`Heng Zhou` is mainly responsible for feature engineering and application parts.

`Zhengqian Xu` is mainly responsible for data exploratory analysis and model building parts.

All of us are engaged in model evaluation, model improvement and results parts， with giving feedback to each other throughout this project.
 
### Motivation
When a customer wants to buy something online, he/she might not be able to input the exact name for that product, and the company needs to figure out which products are most relevant to the search words in order to return the high relevant products to that customer. For example, if a customer wants to buy some “AA battery”, then the website should return “pack of size AA batteries” or at least “cordless drill battery” if they don’t have AA battery, rather than “snow shovel” which is totally irrelevant. 

Based on that scenario, our group first predicted the relevant score between what a user enters (keywords) and what the company has (products) by building supervised machine learning models with NLP techniques. And then on the top of our best model, we developed a mini search recommendation application to return the top 10 high relevant products in the database given a keyword.

The reason why this question is worthy to investigate is that we can help the online website keeps track of customers’ needs. From the historical data analysis, we can figure out which kind of products tend to have low relevance and cannot satisfy our customers very well, so we can introduce new products into the product library. In addition, with an accurate prediction model, the company does not need to manually label relevance score to each search-return pair, and they can use the relevance score to evaluate their recommendation system or search engine. Overall, the more accurately a company can satisfy the customers’ demands; the more business profits/customers a company can gain.
 
### Relevant background research
The original idea for this project comes from a paper [1], in which the author describes the question about matching similar products using their title, description and image. Due to the complexity and lack of data, we choose a similar but easier topic: calculating the relevance score between a searching keyword and a product.

This paper also provides us with some useful hints about how to process the data. For example, the author uses dictionary-based method, conditional random field (CRF) model, CRF with text embeddings to extract features; applies cosine similarity on TF-IDF, paragraph2vec and Silk as products matching baseline; builds random forest, SVM, Naïve Bayes and logistic regression to solve the classification problem. For our project, we refer to the overall procedure discussed in that paper, but use predictive machine learning models. Other papers [2-4] offer helpful guidance about how to calculate similarity between two terms, which give us lots of inspiration.
 
### Data Collection
The data is downloaded from Kaggle (https://www.kaggle.com/c/home-depot-product-search-relevance/data), which contains a number of products and real customer search terms from Home Depot's website. The training data has more than 50,000 rows and each row has a ‘product unique id’, ‘product title’, ‘search terms’ as well as ‘relevance score’. ‘relevance score’ is in the range [1, 3], which measures the relevance between search terms and product. 

There are also other supplementary datasets containing detailed information about each product such as product description, product value, product image and product URL. For this project, we only merge product description and after that the dataset contains more than 70,000 rows. Due to the limitation of speed(it took about 1 hour to finish data cleaning), we randomly select 30,000 observations as a subset from the merged dataset. 
 
### Methods
After data loading, we first cleaned data by removing HTML tags, punctuation and stop words, stemming text, and correcting spelling errors. Then we performed feature engineering by generating new text features. We calculated the length of some features; counted shared, common words between features; computed similarities between features using `fuzzywuzzy`, `cosine similarity` and `jaccard similarity`. 

Next, in the exploratory data analysis, we plotted the `histogram` for all new features to see their distributions, as well as the `heatmap` to see the correlations between them. In the end, we made a `pairplot` for features whose correlation with relevance score is greater than 0.3.

In model building part, we applied four different predictive machine learning models. They are `lasso with cross validation`, `random forest with cross validation`, `xgboost with cross validation`, and `chain models with pipeline` with picking two best models from the previous three models. In the next part. We will discuss the methods in detail.
 
### Procedure
#### Data Cleaning
We mainly dealt with 3 useful columns in this phase: `product description`, `search term`, `product title`. For `product description`, we removed HTML tags, punctuation, stop words, non-alphabetic terms, tokens with length < 2 and stemmelized it. For `search term`, since it is typed from people, it may contain typos and different word classes, so we normalized it and corrected spelling errors. For `product title`, we only normalized it.

#### Feature Engineering 
We created 12 new features in this part, and each of them is closely related to the similarity between `product description`, `search term` and `product title`. The new features are:

length_search: length for `search term`

length_title: length for `product title`

length_description: length for  `product description`

number_of_common_1: the number of common words between token of `search term` and that of `product title`

number_of_common_2: the number of common words between token of `search term` and that of `product description`

number_of_shared_words: the number of shared words between token of `search term`, `product title` and `product description`

ratio_1: number of common words between `search term` and `product title` / length of `search term`

ratio_2: number of common words between `search term` and `product description` / length of `search term`

simple_ratio: simple fuzzywuzzy similarity between `search term` and `product title`

sort_ratio: sorted fuzzywuzzy similarity between `search term` and `product title` without considering the order 

cosine_score: cosine similarity between `search term` and `product title`, which measures the  cosine of the angle between these two term vectors

jaccard_score: jaccard similarity between `search term` and `product title`, which is the size of the intersection divided by the size of the union of these two terms
 
#### Exploratory Data Analysis
Since our goal is to predict `relevance`, we need to identify the correlation between predictor variables and the target. Therefore, we first  plotted the `histogram` for all new features to see their distributions. It is not hard to find that some columns have similar distribution, such as `cosine_score`, `jaccard_score` and `length_description`, as well as `number_of_shared_words`, `simple_ratio` and `sort_ratio`.

Then, when checking the Pearson correlation coefficient of each pair of column in the heatmap, we are surprised to see that the correlation among relevance and all other variables are less than 0.4. What’s more, some features are highly correlated such as `cosine_score` and `jaccard_score`, which we should pay attention to when building linear model.

Next, we picked up variables whose correlation with `relevance` was greater than 0.3, and made a pairplot visualization for them. Unfortunately, we could not find the linear pattern between `relevance’ and other columns, but only some pattern between `jaccard_score` and `cosine_score`, as well as `ratio_1`.

After going through basic evaluation, it may lead to a conclusion that it is not a simple linear regression problem for us if we want to make a perfect model. 

#### Model Building
We randomly selected 70% of our data as training data and the remaining as test data. Then we built  four models to use 12 numerical variables to predict for `relevance`.

Lasso: 

Lasso helps to make an automatic feature selection and we built it as a benchmark. We first tried a range of alpha value and made a trajectory plot of lasso coefficients, showing that `ratio_1` is the last column in the model. Then we used LassoCV to choose the optimal alpha value, which is 1.32*10^-6. Then this value was put into our lasso model and fit the training data. Finally  we got RMSE of test data is 0.487.

Random Forest: 

For this model, we used GridSearchCV to find the best parameters. We set `max_depth` as [5,6,7] and `n_estimators` as [200,400,600] and got the optimal parameters that `max_depth` is 7 and `n_estimators` is 400. It turns out that random forest with best parameters performs better than lasso with test RMSE to be 0.481.

Xgboost:  

For this model, we also used GridSearchCV to find the best parameters. We set `max_depth` as [2,4,6,8] and `n_estimators` as [20,50,100,200] and got the optimal parameters that `max_depth` is 4 and `n_estimators` is 100. It turns out that the result of Xgboost is slightly better than random forest  with test RMSE to be 0.480.

Chain model with pipeline: 

We picked two best models - Random Forest and Xgboost, from the previous three models and built a pipeline to to fit the training data. However, the test RMSE is 0.483, which is higher than both of separating models. 
 
### Results & Analysis
Since we built predictive machine learning models, we used `RMSE` as the evaluation metric. The results are shown in the below table:

![image](https://github.com/zzzzzzhy0607/Natural-Language-Processing/blob/master/Product%20Relevance%20Match/Result.png)

Our best model is Xgboost with test RMSE = 0.48. Compared with the best RMSE score 0.43 in Kaggle competition leaderboard, our result is pretty good since we only trained our model based on a smaller dataset.
 
### Application
On the top of our best model, we developed a mini search recommendation application. We first asked the user to input a search term (i.e. Battery) and then we replaced every search term in the original dataset with the new normalized search term (batteri). Next, we recalculated the 12 similarity measures among the new search term, product title and product description to generate the new prediction dataset and finally applied the trained Xgboost model to the new prediction dataset to predict the relevance score. In the end, we return the user with the top 10 relevance products given the search term he/she entered (i.e. ‘Battery’):

![image](https://github.com/zzzzzzhy0607/Natural-Language-Processing/blob/master/Product%20Relevance%20Match/Search.png)

We can see that most of the products are related to `battery`, but there are still some unrelated products. 
### References
[1] Petar R. Petar P., Peter M., Heiko P.: A Machine Learning Approach for Product Matching and Categorization(2016)

[2] Petrovski, P., Bryl, V., Bizer, C.: Learning regular expressions for the extraction of product attributes from e-commerce microdata (2014)

[3] Ghani, R., Probst, K., Liu, Y., Krema, M., Fano, A.: Text mining for product attribute extraction. ACM SIGKDD Explorations Newsletter 8(1), 41–48 (2006)

[4] Le, Q.V., Mikolov, T.: Distributed representations of sentences and documents. arXiv preprint arXiv:1405.4053 (2014)
 
 
