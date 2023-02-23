from dacota import process_cohortization, get_cohort

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def generate_test_dataframe(task,
                            no_categorical_cols, 
                            no_continous_cols,
                            no_samples,
                           ):
    df = pd.DataFrame([])
    categorical_cols = []
    continous_cols = []
    for j in range(no_categorical_cols):
        feat_name_ = 'var_' + str(j)
        no_categories_ = 2 + np.random.randint(5)
        category_names_ = np.asarray([feat_name_ + '_category_' + str(jj) for jj in range(no_categories_)])
        df[feat_name_] = category_names_[np.random.randint(0, no_categories_, no_samples)]
        categorical_cols.append(feat_name_)
    for j in range(no_continous_cols):
        feat_name_ = 'var_' + str(j)
        df[feat_name_] = np.random.randn(no_samples)*np.random.randn(no_samples)
        continous_cols.append(feat_name_)    
    if(task == 'regression'):
        df['target_col'] = np.random.randn(no_samples)
    else:
        no_categories_ = 1 + np.random.randint(5)
        category_names_ = np.asarray(['target_' + str(jj) for jj in range(no_categories_)])
        df['target_col'] = category_names_[np.random.randint(0, no_categories_, no_samples)]
    return df, categorical_cols, continous_cols

def test_regression(trials=10):
    for trial_ in range(trials):
        print('trial - ', str(trial_))
        task = 'regression'
        no_samples = np.random.randint(1000)
        no_categorical_cols = np.random.randint(10)
        no_continous_cols = np.random.randint(10)
        target_col = 'target_col'
        df, categorical_cols, continous_cols = generate_test_dataframe(task,
                                no_categorical_cols, 
                                no_continous_cols,
                                no_samples,
                               )
        df_cohorts, df_generations = process_cohortization(df,
                    target_col,
                    categorical_cols, 
                    continous_cols,
                    task,
                    minimum_confidence_for_a_cohort = 0.8, 
                    minimum_population_for_a_cohort = 5,
                    maximum_population_for_a_cohort = 25,
                    no_permutations = 10,
                    no_initial_population_size = 100,
                    no_generations = 2,
                    no_random_additions_per_generation = 25,
                    no_mutations_per_generation = 25,
                    no_crossovers_per_generation = 25,
                    top_quantile_to_survive_per_generation = 0.8,
                    no_bins_discretization_continous = 5,
                    )
        if(len(df_cohorts)>0):
            print(df_cohorts.head())
            print(df_generations.head())
            solution_form = df_cohorts.sample(1).solution_form.values[0]
            df_cohort = get_cohort(df, solution_form, continous_cols)
            print(df_cohort.head())
            
            
def test_classification(trials=10):
    for trial_ in range(trials):
        print('trial - ', str(trial_))
        task = 'classification'
        no_samples = np.random.randint(1000)
        no_categorical_cols = np.random.randint(10)
        no_continous_cols = np.random.randint(10)
        target_col = 'target_col'
        df, categorical_cols, continous_cols = generate_test_dataframe(task,
                                no_categorical_cols, 
                                no_continous_cols,
                                no_samples,
                               )
        df_cohorts, df_generations = process_cohortization(df,
                    target_col,
                    categorical_cols, 
                    continous_cols,
                    task,
                    minimum_confidence_for_a_cohort = 0.8, 
                    minimum_population_for_a_cohort = 5,
                    maximum_population_for_a_cohort = 25,
                    no_permutations = 10,
                    no_initial_population_size = 100,
                    no_generations = 2,
                    no_random_additions_per_generation = 25,
                    no_mutations_per_generation = 25,
                    no_crossovers_per_generation = 25,
                    top_quantile_to_survive_per_generation = 0.8,
                    no_bins_discretization_continous = 5,
                    )
        if(len(df_cohorts)>0):
            print(df_cohorts.head())
            print(df_generations.head())
            solution_form = df_cohorts.sample(1).solution_form.values[0]
            df_cohort = get_cohort(df, solution_form, continous_cols)
            print(df_cohort.head())

def test(trials=10):
    test_regression(trials=10)
    test_classification(trials=10)

if __name__ == '__main__':
    test()