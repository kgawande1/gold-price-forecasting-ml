# Group17MLProject

Hi everyone! This is the link to our Machine Learning Project Github Pages.
https://github.gatech.edu/pages/sdesai328/Group17MLProject/

Proposal Video: [Link](https://www.youtube.com/watch?v=OQ27c0DZCqA)


## Part 1: Literature Review and Dataset

In today’s economy, where the market seems very volatile due to inflation and uncertainty, gold
is becoming an increasingly important asset. Being able to retain its value over time, gold is able
to hedge against this volatility making it important for investors to have accurate price
forecasting. Recent studies have explored various machine learning techniques to improve
prediction accuracy. For example, Ghule and Gadhave [1] applied Random Forest regression to
predict future gold rates, illustrating the model’s ability to capture patterns in gold price
movements. In addition, Alharbi [2] also looked into predicting precious metal prices using
machine learning methods from 2017 to 2023, providing insights into the market and predictive
modeling. Finally, Zainal and Mustaffa [3] conducted a literature review on gold price
forecasting techniques, emphasizing the evolution from traditional statistical methods to
advanced machine learning approaches. Their work shows the potential of using multiple
methods to try and forecast the nonlinear dynamics of gold prices which we look to explore.

### Datasets

- a. Historical gold data: [Yahoo Finance](https://finance.yahoo.com/quote/GC=F/)
- b. Economic indicators like inflation, interest rates, GDP growth, and foreign exchange rates [4]: [FRED](https://fred.stlouisfed.org/)
- c. Exchange rates: [IMF](https://www.imf.org/external/np/fin/data/param_rms_mth.aspx)

## Part 2: Problem Definition

Gold prices are highly volatile, influenced by factors like inflation, interest rates, and currency
fluctuations. Traditional forecasting methods struggle to capture nonlinear relationships, leading
to inaccurate predictions. This project aims to develop a machine learning-based approach that
will improve predictive accuracy of these complex interactions by combining supervised
methods for price prediction and unsupervised methods for identifying market patterns. Unlike
single-model approaches, the hybrid method will leverage diverse features and cross-validation
to improve performance.

## Part 3: Methods

### Proposal

For preprocessing methods, we will use data cleaning, data transformation and feature
engineering. Data cleaning will improve data reliability by handling missing values, outliers, and
duplicates. Data transformation like scaling and logarithmic adjustments will standardize
metrics, enabling direct comparisons. Feature engineering will create moving averages, and
volatility indicators to find relationships between gold prices and economic factors.

For ML methods, we will use Random Forest, Linear Regression, and GMM. Random Forest is
useful for financial data due to its ability to capture nonlinear relationships, assess feature
importance, and reduce overfitting by averaging multiple decision trees. Linear Regression
provides a baseline by modeling the relationship between gold prices and macroeconomic
indicators like inflation and interest rates. Finally, GMM will help identify market patterns such
as high and low volatility periods, illustrating structural patterns that influence price movements.

### Final

#### Data Preprocessing Strategies

The first step was to obtain our data using Yahoo Finance and storing gold prices from the ticker (GC=F) as a pandas series where we have Gold Open, Gold Close, and Gold Volume. After getting the data, we indexed it by date and forward filled any missing values. Then, we decided to do feature engineering to create features that could explain price movement over time. This included adding a 5 day moving average, 100 day moving average for the Gold Close prices, and 5 day volatility. In addition, we take more data from FRED using its API. We take quarterly information like GDP, Interest Rates, CPI, and Unemployment. Since it is quarterly data, we use the same number for all dates in its specific quarter. Additionally, we take foreign exchange rates as more feeatures, including the exchange rate USD-CHF. After collecting all the economic indicators, we converted them into DataFrames, merged them into a single dataset, and reindexed everything to a daily frequency with forward filling to handle missing values. We also applied a log transformation to GDP to reduce skewness. Finally, we combined this economic data with the gold prices, calculated daily gold price changes, and saved the final dataset for modeling.

#### Supervised and Unsupervised Algorithms

As a major change from our initial proposal, we decided to change the algorithms we will implement since the initial attempts produced high MSE error on testing data versus training data. We chose to implement one unsupervised and three supervised learning model to classify our data into price predictions. 
For our unsupervised learning model, we are using KMeans, and for our supervised learning model, we are using Ridge Regression, Neural Network and Random Forest.

Supervised Learning Model 1 (Ridge Regression): For our supervised learning method, we used Ridge Regression to forecast a specific daily Gold Close Price. We used ridge regression to add L2 regularization to prevent overfitting, especially with features that are very highly correlated, such as the USD/CHF and SP500. We split the data into 80% training and 20% testing to lead to the prediction of prices. We chose the ridge regression model because we wanted to prevent overfitting, so it can be more robust in predicting volatile markets. Additionally, the model handles multicollinearity well, making it suitable for financial datasets where predictor variables often move together, ensuring more stable and interpretable coefficient estimates.

Supervised Learning Model 2 (Neural Network): For our second supervised learning model, we created a neural network to forecast daily Gold Close prices. We used this model to capture the complex relationships between the various macroeconomic indicators, including interest rates, unemployment and foreign exchange rates. Additionally, we also utilized technical signals like RSI and the gold prices moving averages. We used two hidden layers one with 64, the other with 32 neurons, ReLU activation functions, and dropout layers to prevent overfitting of the data. We also implemented an Adam optimizer and early stopping based on validation loss. This allowed us to avoid any unecessary training once the model plateaued. Overall, the neural network approach allowed us to capture subtle patterns in the financial time series data, improving prediction performance in certain scenarios.

Supervised Learning Model 3 (Random Forest): For our third supervised learning model, we made daily Gold Close price predictions with Random Forest Regressor. The goal of this model was developing diverse predictions based on our features and data. The two processes used are bootstrapping which creates different subsets of training data and feature sampling where only a random set of features is used in each decision tree. Thus, we create diversity in our models that look to capture non-linearities. This ensemble learning strategy attempts to reduce overfitting while introducing noise and variation in financial data. To validate our results, we used Five-Fold Cross Validation approach. Using metrics such as Mean Squared Error and $R^2$ across folds, we can compare this model to the others. Finally, we visualize the results of the model in a plot against the original data. 

Unsupervised Learning Model 1 (KMeans):
K-means allows us to cluster similar data points that represent a common set of market trends. This can help us identify patterns in the gold market without using predefined labels. We implemented K-means with 4 clusters to group data into 4 types of market trends: bullish gold markets, bearish gold markets, stable markets, and high volatility periods each representing some commonalities between the features. K-Means makes sense to find these patterns in gold commodity behavior and cluster our data into these distinct categories. K-means alllows us to also deal with the high-dimentional data and organize it into these meaningful clusters. In the next section, we will talk more about what our clusters signify and justify what specific sets of feature values correspond to each of these clusters.


## Part 4: Results and Discussion

### Proposal

We will evaluate model performance using R², Mean Absolute Percentage Error (MAPE), and
k-fold cross-validation. R² measures the proportion of variance in gold prices captured by our
model, with higher values indicating better predictive accuracy. MAPE quantifies the average
prediction error in percentage terms, essential for assessing financial impacts. K-fold
cross-validation ensures robust performance by reducing overfitting and improving
generalization across data segments.

Our project aims to predict gold prices using market data, economic indicators, commodity
futures, and exchange rates. To improve prediction accuracy we will combine Random Forest
and Linear Regression for supervised learning with GMM for market pattern identification. We
prioritize sustainability and ethics by promoting accurate predictions to reduce speculative
mining and ensuring transparency in feature importance to mitigate data bias. Our expected
result is the predicted price of gold, derived from multiple features such as interest rates,
inflation, and exchange rates, alongside a high R² score, low MAPE, and consistent performance
validated through cross-validation, which will provide insights to future investors into future
market trends.

### Final

## Initial Results

After implementing the first models, we saw that there was high correlation between features such as the exchange rates and CPI in relation to GDP. This made us reduce the number of features so that one exchange rate and log GDP would become significant in our model. This also reduced our MSE in training data and testing data. We initially used a 80/20 split for training data and testing data but we wanted to make sure that time wasn't as impactful on the model. Thus, we implemented a K-Fold approach which tested our data into 5 testing sections and showed meaningful results.

## Ridge Regression w/K-fold Cross Validation Plots and Metrics

#### Model + Quantitative Metrics

![ridge_regression_model](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/0da553d6-297e-4264-8f32-e47b51995ea6)

![ridge_regression_residuals](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/792db509-94f4-4014-8722-02cb9bbc1d94)

![Screenshot 2025-03-31 at 8 47 26 PM](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/323ef232-886f-454f-a4f8-7f1463e1861c)

#### Ridge Regression Model Evaluation + Discussion
Though on first glance, the model may seem overfitted, as the predicted gold close line matches the true gold close to a striking degree, we believe that our model is not overfitted. Models that suffer from severe overfitting exhibit a very high Training $R^2$ and a much lower Testing $R^2$. This is becasue overfitted models generalize poorly to unseen data, resulting in a low R-squared on the test data. However, our model doesn't suffer from this same plight. With a Testing $R^2$ of 0.9998 and a Testing $R^2$ of 0.9983, the model performs very strongly on both the training set and testing set, showing that it generalizes nicely. Additionally, we can say the same about the Mean Squared Error Values. Both the training and testing set exhibit similar MSE values (~52), showing that we indeed are not overfitting. Thus, based on the strong metrics, we can be pretty confident in our model and its ability to predict gold prices.

## KMeans 

#### Determining Number of Groups

![kmeans_cluster_elbows](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/aff7786f-1406-4ea8-b0a2-249e6f759800)

As shown by the plot above, once we reach 4 groups, there seems to be an elbow. The point suggests diminishing returns from adding more clusters, thus we chose to proceed with 4 clusters.

#### KMeans Model

![kmeans_image](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/5ebce3cc-5025-40dc-9917-be61d8d7f3b1)

![Screenshot 2025-03-31 at 9 25 12 PM](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/69fcec71-7700-4c84-854d-b2ae1106bf1e)

##### Group Classifications: 
- Cluster 0 (Purple): Bullish Gold Market (Rising gold prices, high RSI, moderate economic uncertainty)
- Cluster 1 (Blue): Bearish Gold Market (Falling gold prices, low RSI, strong stock market performance)
- Cluster 2 (Teal): Stable Gold Market (Flat prices, low volume, low volatility, economic stability)
- Cluster 3 (Yellow): Gold Boom / High Volatility (Strong price surges, high trading volume, economic event-driven spikes)

#### KMeans Model Discussion
Using KMeans allows us to segement the many features of the data into 4 distinct groups. This enables us to look at any day in the dataset, or in the future, and label it into one of these 4 groups, giving us an insight into how the gold price may move that day. However, it does seem that KMean is relatively hit-or-miss when it comes to evaluating our dataset. Some clusters are well-separated, while others have overlapping regions, indicating that K-Means might not have perfectly segmented the data. Regardless, this does give us some valuable insight in how the price of gold may change on a certain day, given other measures.

## Neural Network

#### Explanation
With the neural network, he coped to capitalize on several of its advantages as a predictive model. We had previously experimented with Random Forest, but found that it could not account for prices outside of the range of training data. This was not the case with the neural netowrk as it seemed to be a more adaptable model, that could accurately predict market trends regardless of the range of training or testing data.  

#### Model and Visualizations
![True vs Predicted Prices](https://github.gatech.edu/sdesai328/Group17MLProject/assets/78113/d57190d2-8997-4dcd-b69e-0e6fb1ba8936)

![Residual Plot](https://github.gatech.edu/sdesai328/Group17MLProject/assets/78113/5746aef7-846b-4f1e-9581-651737f97d0e)

We acheived these metrics:
```
Training MSE: 1472.4657
Test MSE: 1093.5421
Training R²: 0.9930
Test R²: 0.9888
```

#### Neural Network Discussion
We can see from our visualizations that the neural network performs well in terms of analyzing different macroeconomic indicators and predicting the complex relationships between the many variables that affect gold pricing. However, this model also has its limitations. We can see from the residual plot that the predictions generally stay within close range, but the model is susceptive to slight overpredictions or underpredictions particularly toward the more end of the time range.

Based on our training and testing metrics, we can see that our training and test MSE are in a decent range, where our testing error is actually lower. Our $R^2$ values indicate that the model explains the vast majority of the variance in the data it is fed. Overall, our metrics show strong generalization and high accuracy, though there is still some room for improvement at extreme price points.

## Random Forest

#### Model + Quantitative Metrics
![image](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/c4ad86ac-cfd9-4575-b5f6-38bc8df96286)

![image](https://github.gatech.edu/sdesai328/Group17MLProject/assets/70457/fde1c6dd-0a3a-4aa4-9fd2-f1b22b11a8c8)

We achieved these metrics:
```
Fold 1 - MSE: 75537.47, R²: -1.76
Fold 2 - MSE: 232594.56, R²: -1.46
Fold 3 - MSE: 2889.65, R²: 0.84
Fold 4 - MSE: 2330.88, R²: 0.96
Fold 5 - MSE: 116638.69, R²: -0.06
Average MSE: 85998.25
Average R²: -0.30
```

#### Random Forest Discussion
Looking at the Random forest vizualization and results, we can see the results are pretty skewed. At first, for the beginning few years, the model does pretty well at predicting the prices before 2024 as the predictions follow the actual prices. However past 2024, the predictions are unable to forecast the upward trend that occurs with gold prices as it continues to predict prices that have a similar value as previous years. This is likely because since the training data, being split before these recent years, only contained values that capped around the 1800-1900 level. This made it so the decision trees were unable to grow past values that exceeded those levels, making the predictions unable to reach the level of 2000-3000+ which prices are currently at. This lead to most of the forecasts to still follow some of the similar spikes and falls that the actual prices did but were unable to reach the prices levels that the actual prices hit after 2024. As a result, the predictions outputted by the model became exceedingly poor, as shown by the negative $R^2$ values in folds 1, 2, and 5.

## Next Steps

With regard to our KMeans implementation, we would like to perform a Principal Component Analysis (PCA) on our dataset to reduce the dimensionality of our data. Since our dataset contains many featues, and some being highly correlated, it would benefit us a lot to "simplify" the data via dimensionality reduction. This would likely help our clustering perform even better, as higher dimensionality is known to cause clustering algorithms to perform poorly. PCA will help us by removing noise and focusing on the most important patterns. We could also explore a probabilistic view of our data, using a Gaussian Mixture Model (GMM) to visualize and classify our data. This is because the stock market and price movements of commodities can be stochastic in nature, with normal distributions underlying many models. Using a GMM will allow for flexible cluster shapes and probabilistic assignments, which is known to be useful when cluster boundaries are not clear-cut or data is complex, as such is our case. 

Another methodology we could try and explore would be Logistic Regression, both Binary Classification and Multiclass Classification. In the binary model, the 2 categories would be "price increases" and "price decreases". In the multiclass model, we would have more nuanced categories, such as "large increase: price increases by over 5%", "small increase: price increases between 2-5%", "small/no chance: price stays within 2% of previous price", "small decrease: price decreases between 2-5%", and "large decrease: price decreases by over 5%". This would allow us to bucket our data/days into categories depending on how gold performs that day, then reverse engineer and see what common traits these days share.

## Part 5: References

- [1] R. Ghule and A. Gadhave, "Gold Price Prediction using Machine Learning,"
International Journal of Scientific Research in Engineering and Management, vol. 6, no.
7, pp. 1-5, 2022. [ResearchGate](https://www.researchgate.net/publication/362491642_Gold_Price_Prediction_using_Machine_Learning)

- [2] M. H. Alharbi, "Applying Machine Learning Techniques to Analyze and Explore
Precious Metals," Technology and Investment, vol. 15, no. 4, pp. 183-197, 2024. [SCIRP](https://www.scirp.org/journal/paperinformation?paperid=136582)

- [3] N. Zainal and Z. Mustaffa, "A Literature Review on Gold Price Predictive
Techniques," in Proceedings of the 2015 International Conference on Advanced
Mechatronic Systems, Beijing, China, 2015, pp. 252-257. [IEEE Xplore](https://ieeexplore.ieee.org/document/7324301)

- [4] G. K. Nair, N. Choudhary, and H. Purohit, "The Relationship between Gold Prices and Exchange Value of US Dollar in India," Emerging Markets Journal, vol. 5, no. 1, pp. 1–7, 2015. [Online](https://emaj.pitt.edu/ojs/emaj/article/view/66)

## Part 6: Contribution Table and Gantt Chart

[Google Sheets Gantt Chart](https://docs.google.com/spreadsheets/u/0/d/1NS2EDKeplf-mkVPEBQIdRBL6Zhtu8XMwBucQmdk5YFE/edit)

### Proposal

| Name     | Proposal Contributions |
|----------|------------------------|
| Aiden Wu | Contributed to the Literature Review, Background and Introduction, and Problem Definition. |
| Krish Gawande  | Contributed to the Github Pages, the Potential Results and Discussions, and Gantt Chart.          |
| Samay Desai  | Contributed to the Gantt Chart and the Potential Results and Discussions.           |
| Jason Li  | Contributed to the ML Methods to use and the Potential Dataset section.          |
| Liam Dolphin  | Contributed to the ML Methods to use and the Potential Dataset section.          |


---
### Midterm

| Name   | Midterm Contributions |
|--------|------------------------|
| Aidan  | Contributed to Model 1 Design & Selection, Model 1 Data Cleaning, Model 2 Design & Selection, Model 2 Data Cleaning, Model 1 and Model 2 Results Evaluation, and Midterm Report. |
| Liam   | Contributed to Model 1 Design & Selection, Model 1 Data Visualization, Model 2 Data Visualization, Model 1 and Model 2 Results Evaluation, and Midterm Report. |
| Jason  | Contributed to Model 1 Data Cleaning, Model 1 Feature Reduction, Model 2 Feature Reduction, Model 1 and Model 2 Results Evaluation, and Midterm Report. |
| Krish  | Contributed to Model 1 Feature Reduction, Model 2 Design & Selection, Model 2 Coding & Implementation, Model 1 and Model 2 Results Evaluation, and Midterm Report. |
| Samay  | Contributed to Model 1 Implementation & Coding, Model 2 Data Cleaning, Model 2 Feature Reduction, Model 1 and Model 2 Results Evaluation, and   Report. |

---
### Final

| Name   | Final Contributions |
|--------|---------------------|
| Aidan  | Contributed to Model 3 Design & Selection, Model 3 Data Cleaning, Model 3 Results Evaluation, M1-M3 Comparison, and Final Report. |
| Jason  | Contributed to Model 3 Design & Selection, Model 3 Feature Reduction, Model 3 Results Evaluation, M1-M3 Comparison, and Final Report. |
| Krish  | Contributed to Model 3 Data Cleaning, Model 3 Implementation & Coding, Model 3 Results Evaluation, M1-M3 Comparison, and Final Report. |
| Liam   | Contributed to Model 3 Data Visualization, Model 3 Results Evaluation, M1-M3 Comparison, and Final Report. |
| Samay  | Contributed to Model 3 Implementation & Coding, Model 3 Results Evaluation, M1-M3 Comparison, and Final Report. |


