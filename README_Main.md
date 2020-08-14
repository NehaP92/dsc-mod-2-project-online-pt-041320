# Mod 2 Final Project: King County Data Analysis



## Introduction

A lot of us today wanting to sell our homes invest loads of time researching the market and the trends before setting a price for their homes, but at times, end up selling at a price below expectation. It becomes stressfull at times, because real estate, is a big investment one expects a desent return from. However, just knowing the market is not enough. Certain features of a house play significant role in increasing the cost of your house to just the bar you need. Some of these features do not require particular upfron investment either. The findings of this project would help to identify some of the features worth considering to set and expect a price for your house.


## Dataset

The data was extracted from the `'kc_house_data.csv'`. The dataset contains the following information:
- **id** - unique identified for a house
- **dateDate** - house was sold
- **pricePrice** - is prediction target
- **bedroomsNumber** - of Bedrooms/House
- **bathroomsNumber** - of bathrooms/bedrooms
- **sqft_livingsquare** - footage of the home
- **sqft_lotsquare** - footage of the lot
- **floorsTotal** - floors (levels) in house
- **waterfront** - House which has a view to a waterfront
- **view** - Has been viewed
- **condition** - How good the condition is ( Overall )
- **grade** - overall grade given to the housing unit, based on King County grading system
- **sqft_above** - square footage of house apart from basement
- **sqft_basement** - square footage of the basement
- **yr_built** - Built Year
- **yr_renovated** - Year when house was renovated
- **zipcode** - zip
- **lat** - Latitude coordinate
- **long** - Longitude coordinate
- **sqft_living15** - The square footage of interior housing living space for the nearest 15 neighbors
- **sqft_lot15** - The square footage of the land lots of the nearest 15 neighbors


## Methodology

We begin by exploring the data set, and understanding the raw features that are provided. Next, it is important to understand the null values and their potential implications towards the model, then taking the necessary steps to eliminate these.

Each column of the dataset is explored and analysed to aid in the initial decision of weather to keep or remove the columns. Further, based on the same analysis results, the remaining columns are then bifurcated into numerical or caegorical categories.

The Numerical columns are then further analysed for multicolinearity, with the cut-off of 0.75, while the Categorical columns are converted using the one-hot encoding method. 1st columns of the resulting encoded categorical columns are dropped to avoid errors due to multicollinearity.

A base model is made from the final datatable thus created. Coefficients, pvalues, normality, and hetroscedasticity are then analysed to help formulating the corrective steps to improve the model. `statsmodels` is used to fit the data using OLS methodology.

After creating a few models, the data is then checked for outliers and divided into two groups. One (Group1), pertaining to the majority of the dataset, and another, to target the pricy houses (Group2). Group1 was simply extracted by removing the outliers using IQR from the main dataset, while to generate optimum results with Group2, these were selected based on the visual trial and error method. Seperate models were created for both these groups and the coeffs analysed.

The dataset was also divided on the basis of having a basement or not, for group 1, since this division resulted in better model fit.

Once the models were created, seperate plots for the each determing feature for each model were created, along with one common error plot for all the coeffecients together since each individual feature behaves differently when combined with all the features as compared to independent influence on the dependent variable.


## Python Functions Created for This Project

1. Generate a model using `statsmodels` module's `ols` method

```
def ols_model(cat_col, num_col, df):
    """Generates an OLS model fit for the given dataframe.
    Input:
    cat_col (list): list of columns in the dataframe that are categorical
    num_col (list): list of columns in the dataframe that are numerical
    df (DataFrame)
    Output:
    model object from statsmodel.ols"""
    cat_col_form = [f"C({col})" for col in cat_col]
    num_predictors = '+'.join(num_col)
    col_predictors = '+'.join(cat_col_form)
    formula = 'price~' + num_predictors + '+' + col_predictors
    model = ols(formula = formula, data = df).fit()
    return model
```
    
2. Conduct basic model analysis for coeffecients, pvalues, and homoscedasticity

```
def model_analysis(model,df):
    """takes the model and data frame and returns the model coeffecients, 
    evaluates p-values based on alpha 0.05, checks for normality through Q-Q plot,
    and analyses the homoscedasticity of our data
    input:
    model (OLS model)
    df (DataFrame): the DataFrame on wchich the analysis is done
    """
    residual = model.resid
    price = df['price']
    sns.scatterplot(price, residual);
    plt.axhline(0)
    display(model.params.to_frame().style.background_gradient())
    display(model.pvalues<0.5)
    display(sm.graphics.qqplot(model.resid, stats.norm, line='45', fit = True));
```
    
3. Determine the 1st quartile, 3rd quartile and IQR of a series

```def IQR(df,category):
    """Gives the IQR output for a given series in the dataframe
    Input:
    df (DataFrame)
    category (str): name of the column"""
    Q1 = df[category].quantile(0.25)
    Q3 = df[category].quantile(0.75)
    IQR = Q3-Q1
    return Q1,Q3,IQR
```

4. Remove outliers based on IQR

```def IQR_remove_outlier(df, col):
    """Removes outlier based on IQR"""
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3-Q1
    group = df[~((df[col]<(Q1-1.5*IQR)) | (df[col]>(Q3+1.5*IQR)))]
    return group
```
    
5. Create a `coefplot` for the regression model

```def feature_err_plot (model):
    """Creates a coefficient plot for the regression model
    Input:
    model (statsmodels OLS model object)
    Output:
    matplotlib coeffplot
    """
    err_series = model.params - model.conf_int()[0]
    coef_df = pd.DataFrame({'coef': model.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]
                       })
    
    fig, ax = plt.subplots(figsize=(15, 8))
    coef_df.plot(x='varname', y='coef', kind='bar', ax=ax, color='none', yerr='err', legend=False)
    ax.set_ylabel('Price')
    ax.set_xlabel('Features')
    ax.scatter(x=pd.np.arange(coef_df.shape[0]), marker='s', s=120, y=coef_df['coef'], color='c')
    ax.axhline(y=0, linestyle='--', color='midnightblue', linewidth=2)
    ax.xaxis.set_ticks_position('none')
    _ = ax.set_xticklabels(coef_df.varname, rotation=90, fontsize=12)
    plt.show()
```


## The Base Model

### Exploring and Scrubbing individual columns

The first step to any modelling process is to examine and analyze the given data. An initial cleaning of the data set included visual representation of the null values of the dataset using `missingno.matrix()` method.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/null_values_1.png' width=800>

This combined with numerical analysis of the null values show that the data has about 11% null values in the `waterfront` column and 18% in `yr_renovated` column. This is a large number to risk the elimanation of the entire row, hence, we assume that the null values indicate no information and replace those with `False`.

Further, the following edits were run through the columns:
- The date dtype was converted to `DateTime` to make it easier to strip just the year (`year`) a house was sold and a new column `age` was added by subtracting `year` from `date`. Columns `year` and `date` were then dropped to avoid multicolinearity.
- `id` column is removed since it is randomnly assigned numbers to recognize the house and should bear no linear relation with the price.

#### Sorting into Numerical and Categorical Types
A basic analysis was done on all the columns to determine whether that column should be categorical or numerical. This analysis was conducted by evaluating the results of `pandas.Series.describe()` and `pandas.Series.unique()` methods to identify the similiraties of a colum with categorical data types. Based on the initial analysis, the categorical and numerial columns were as below:
```
categorical_columns = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade', 'date_year']
```
```
numerical_columns = ['sqft_living','sqft_lot','sqft_above','sqft_basement','yr_built','yr_renovated',
'zipcode','lat','long','sqft_living15','sqft_lot15','age']
```


### Multicolinearity Analysis
It is recomended that multicolinearity test be done only within the numerical data. However, for the base model, all the columns of the initial raw dataframe were tested for multicolinearity for easy elimination. `pandas.DataFrame.corr()` method was used the results of which were plotted on a `seaborn` heatmap.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/multicollinearity_initial.png' width=800>

The cut off rule selected is 0.75, with which the columns `sqft_above`, and `sqft_living` were eliminated. The relation with dependent variable `price` was also considered before selecting these columns to drop.


### The Base Model
The base model was generated by OLS fitting of the data set through `statsmodels`'s `old` method. The categorical and numerical data considered for this model are described below:

```
df1_categorical = ['floors', 'waterfront', 'view', 'condition']
```
```
num_columns = ['sqft_living', 'sqft_lot','sqft_basement','zipcode',
'lat', 'long', 'sqft_lot15', 'age','grade','bedrooms', 'bathrooms']
```

In order to aquire interpretable and comparable results, the numerical data is first scaled using `sklearn.preprocessing` module's `RobustScaler()` class. The categorical data is also transformed by one hot encoding method using `pandas.get_dummies()`. The first (base) model is then generated using the `ols_model()` function, specifically created for this project.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/model_results_base.png'>

The first model yeilded an R squared of 0.703. Further analysis on the validity of this model, using the `model_analysis()` function specifically created for this project shows that the data is not normal and there seems to be large number of outliers on the higher end. THis gives us an oportunity to divide our data set into two groups, the one without the outliers, and by not completely disregarding the outliers, the other group which would only have the higher end values. This is done later in the later sections. The homoscedasticity test also shows large hetroscedasticities, especially on the higher side.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/base_QQ.png'>

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/base_hom.png'>

To improve the fit of this model, a few changes were made to certain columns.


### Working on Other Columns

- `yr_renovated` was converted to categorical form indicating whether the house was renovated or not. All the renovations after the house was sold, and the `NaN` values were considered as not renovated. 
- `zipcodes` were divided into 8 zones/subregions based on the data given on government website and grouped as categorical column. `zipcodes`, `lat`, and `long` were removed from the data to avoid running into multicollinearity errors.
- `sqft_basement` converted to categorical with booleans if the house has a basement or not. `NaN` values were assumed as `False`.

The resulting model improved the R sqaured to 0.722.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/model_results_2.png'>

The normality and homoscedasticity tests gave the same results since the outliers were not worked upon. The following section goes into removing the outliers and dividing our data into two groups.


## Spliting into Two Groups

Distribution plots play an important role to visually identify the normality and skewness of our data. Since our aim is to split the group based on prices, the ependent variable, we would first have a look at the distribution of `price`.

### Group 1

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/price_1.png'>

The dataframe final data frame is stripped off outliers using the `IQR_remove_outlier()` function created specifically for this project.

Other columns are also checked to see if this improved the normality to an extent. however, we see that `sqft_lot` and `sqft_lot15` are still massively skewed.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/all_dist.png' width=800>

Once again, we remove the outliers, now on the basis of `sqft_lot` column, and visualise the normality.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/sqft_lot_1.png'>

The ols model is now fitted on this data set, and the R sqauared resulted in 0.743

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/group1_model.png'>

You may also see the improvemet in the Q-Q plot and the improved homoscedasticity result:

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/group1_analysis.png'>

#### Further splitting based on Basement

`Group1` was also further split in two, with and without basements. The results showed a better R squared fit at 0.76 for houses without basement.


### What happens to the outliers?

Since, there are a lot of outliers in the higher side, there may be another set of data that could fit a linear regression model, of all the pricy houses. To begin with, all the higher end outliers were chosen and the outliers were first removed from the `sqft_lot` since it had significantly large skewness. A model was fit into this data, which resulted in a lower R sqare value of 0.503.

To identify the best set of datapoints, the distribution plot was generated, and the next set of values were chosen based on the visual trial and error method. The heighest R square of 0.784 was acheived with the houses priced at above $2,500,000.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/group2_model.png'>

However, a lot of features didnt satisfy this model, and were dropped out. The final model, resulted in an R sqare of 0.746.

<img src='https://raw.githubusercontent.com/NehaP92/dsc-mod-2-project-online-pt-041320/master/group2_model2.png'>


## Findings and Conclusions

All analysis and interpretations were made based on the feature coeffecients, since they indicate how much a particular feature would positively or negatively affect the dependent variable, here, `price`. Individual plots were then made to understand the relationshop of the most influential features with price of the house.

It is important note here that the calculated influence indicates the effect of a particular feature **combined** with all other features, and individual analysis **MAY NOT BE THE SAME**.

### Main Model

Based on the coeffecients calculated through the ols model fit, the base value of a house, with none of the features were sold at $380,036

The heighest positive influence were seen to be `waterfront`, `view`, and `sqft_living`, while the heighest negtive influencing features include a few locations, `floors`, and `basement`


### Group1

Based on the coeffecients calculated through the ols model fit, the base value of a house, with none of the features were sold at $309,739

The heighest positive influence were seen to be `waterfront`, `view`, `condition`, and houses in the `east_urban` region of king county while the heighest negtive influencing features remain certain locations

While splitting into groups based on basement, the base value of the house without any features show houses without basements to sell at a higher price of $335,820, compared to $119,992 for houses with basement. Further analysing the houses with basement shows an inverse relation with price. This is because, smaller basements are likely to be garden basements which are expensive to build (also depends on the soil), while liveable basements which are larger in size have the area included inthe sqft living and drops the cost considerabely.


### Group2

Based on the coeffecients calculated through the ols model fit, the base value of a house, with none of the features were sold at $796,597

The heighest positive influence were seen to be certain locations, renovations, and `condition`, and houses in the `east_urban` region of king county while the heighest negtive influencing features remain certain locations. It is always seen that pricy homes are located at only certain areas of the county, where either the prices rise due to the land or the location. When people buy expensive houses, they really want to invest into a house that is in a good condition and wont require them of further cost, and also consider the value for money. Renovations are likely to influence the price since, most of the renovated pricy homes have a mordern sophesticated interiors which cost significantly high.


## Recommendations