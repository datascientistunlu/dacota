!docs/assets/img/dacoto_docs_01.png

# ------

# **DACOTA**

### **Overview**

#### “Dacota”, Datategy Cohort Targeting, (pronounced /- dəˈkoʊtə/ ) is a python library developed by Datategy SAS’s data science team making automatic cohort discovery a breeze.  We have identified the crucial need and urgency among the community to provide a tool which identifies cohorts related to a target variable (subgroups of the dataset of various sizes with similar characteristics) with solid statistical guarantees. 

#### Using Dacota, by choosing your ultimate task, whether regression or classification and your desired parameters related to cohortization and optimization process, with a single click you can identify the subgroups with statistical confidence.


### **Cohort Centric Data Science and ML**

#### Ever since Andrew Ng coined the term “Data Centric AI”, the community has increased its efforts to focus more on data in terms of analysis, feature engineering and quality augmentation rather than ML models. Though this observation is totally correct especially in terms of tabular tasks, there exists an important gap, especially for certain types of data and context : The need for a cohort centric approach. 

#### When considering “imbalance” in data science and ML, we directly refer to the imbalance of target variable’s distribution, either for regression or classification. From a perspective, this stems from the fact that we generally want to optimize an “overall metric”, averaged over whole test dataset such as “mean squared error” or “mean classification accuracy”. However, most of the time, in real life use cases, overall metric is far from reflecting the actual objectives. This is due to the “imbalance in predictors” rather than on target variable. In other words, due to the cohortized nature of the dataset. 

#### These cohort specific concerns are prevalent especially in sensitive ML use cases, such as healthcare, security, finance etc. For instance, a dataset for developing a medical treatment may contain mostly young individuals, where the ML model optimized to maximize mean accuracy would discard the performance among elders, a cohort of minority. Of course, in this specific case the real life objective is to retain a certain degree of reliability for any important subgroup, rather than the overall metric. 

#### Though there exists many post hoc set of ML model performance assessment techniques, the cohort awareness should exists since the beginning of the development cycle, in exploratory data analysis.  Therefore, as Datategy data science team, we wanted to declare a manifest of “Cohort Centric AI”. 

#### The library is at its extreme infancy, probably requiring tons of improvements in computation and also many functional extensions. We just wanted to spark an initiative around our cohort centric manifest and encourage community to improve the library. In our seed version, considering practically infinite set of options when it comes to automatic dataset cohortization, we aimed an optimal trade-off of usability, complexity,  customizability, speed, statistical reliability and accuracy. 

### **How it works ?**

#### Dacota uses genetic algorithm to identify diverse set of cohorts inside a dataset for a particular task related to a target variable.  In our devised framework, we assume each predictor is categorical. For continuous predictors, the algorithm automatically discretizes them in quantiles with number of bins set by the user. Given this constraint, a cohort can be encoded as a binary string like “10110…”, each bit representing a category of an input variable. Obviously, a cohort for a single feature cannot be all ‘0s’, whilst it can be all ‘1s’ (indicating this cohort for this particular feature has no statistical constraint) and any other combination. 

#### The idea is to identify cohorts which are divergent from the whole dataset with predetermined minimum statistical confidence. For this purpose, in case of regression, we perform permuted t-test with unequal variance. Fitness of a cohort in this case represents the absolute  distance of cohort’s target variable mean from the whole dataset. For classification, we perform a permuted version of Jensen-Shannon distance measurement of categorical distribution of target variable. To incorporate the effect of cohort size, for each permutation, we sample a random control group of the same cohort size, record the JS distances and perform a t-test among sampled distances.   

#### Under this context, genetic algorithm is the natural choice of optimization as it provides maximum diversity in produced results and the notions of mutation and crossover fits well to the concept of cohortization.

#### The user can choose the minimum and maximum size of his/her cohort definition and minimum statistical confidence among other variables related to genetic algorithm optimization.

### **Usage**

```python
## classification

from sklearn.datasets import load_wine
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from dacota import process_cohortization, get_cohort

dict_data = load_wine()
df = pd.DataFrame(data = dict_data['data'] , columns= dict_data['feature_names'])
df['wine_class'] = dict_data['target'].astype('str')
del(dict_data)

target_col = 'wine_class'
task = 'classification'
categorical_cols = []
continous_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines', 'proline']

df_cohorts, df_generations = process_cohortization(df.fillna(0.0),
                    target_col,
                    categorical_cols, 
                    continous_cols,
                    task,
                    minimum_confidence_for_a_cohort = 0.90, 
                    minimum_population_for_a_cohort = 5,
                    maximum_population_for_a_cohort = 30,
                    no_permutations = 100,
                    no_initial_population_size = 5000,
                    no_generations = 20,
                    no_random_additions_per_generation = 1000,
                    no_mutations_per_generation = 1000,
                    no_crossovers_per_generation = 1000,
                    top_quantile_to_survive_per_generation = 0.75,
                    no_bins_discretization_continous = 5,
                    )

solution_form = df_cohorts.sample(1).solution_form.values[0]
get_cohort(df, solution_form, continous_cols)
```

```python
## regression

from sklearn.datasets import load_wine
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from dacota import process_cohortization, get_cohort

dict_data = load_wine()
df = pd.DataFrame(data = dict_data['data'] , columns= dict_data['feature_names'])
df['wine_class'] = dict_data['target'].astype('str')
del(dict_data)

target_col = 'proline'
task = 'regression'
categorical_cols = ['wine_class']
continous_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
       'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
       'proanthocyanins', 'color_intensity', 'hue',
       'od280/od315_of_diluted_wines']

df.loc[:, target_col] = df[target_col].astype('float').fillna(0)
df.loc[:, continous_cols] = df[continous_cols].astype('float').fillna(0)

df_cohorts, df_generations = process_cohortization(df.fillna(0.0),
                    target_col,
                    categorical_cols, 
                    continous_cols,
                    task,
                    minimum_confidence_for_a_cohort = 0.90, 
                    minimum_population_for_a_cohort = 5,
                    maximum_population_for_a_cohort = 30,
                    no_permutations = 100,
                    no_initial_population_size = 5000,
                    no_generations = 20,
                    no_random_additions_per_generation = 1000,
                    no_mutations_per_generation = 1000,
                    no_crossovers_per_generation = 1000,
                    top_quantile_to_survive_per_generation = 0.75,
                    no_bins_discretization_continous = 5,
                    )

solution_form = df_cohorts.sample(1).solution_form.values[0]
get_cohort(df, solution_form, continous_cols)
```