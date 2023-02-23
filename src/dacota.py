import pandas as pd
import numpy as np
import copy
import random
import scipy.stats
import math
from itertools import groupby
import time
from scipy.spatial.distance import jensenshannon

import warnings
warnings.filterwarnings("ignore")

MAX_TRIALS_NEW_AUTHENTIC_SOLUTIONS = 100
INVALID_COHORT_FITNESS = 0.0






def get_category_frequencies(df, target_col, target_list_classes, normalize=False):
    dist_ = df[target_col].value_counts(normalize=normalize).reindex(target_list_classes, fill_value=0.0)
    return dist_

def get_feature_category_definitions(df, input_features, task, target_col, 
                                     categorical_cols, continous_cols,
                                     minimum_confidence_for_a_cohort, 
                                     minimum_population_for_a_cohort,
                                     maximum_population_for_a_cohort,
                                     no_permutations,
                                    no_initial_population_size,
                                    no_generations,
                                    no_random_additions_per_generation,
                                    no_mutations_per_generation,
                                    no_crossovers_per_generation,
                                    top_quantile_to_survive_per_generation,
                                     no_bins_discretization,
                                    ):
    dict_feature_category_definitions = {}
    dict_feature_category_definitions.update({"input_features":input_features})
    dict_feature_category_definitions.update({"no_input_features":len(input_features)})
    dict_feature_category_definitions.update({"set_input_features":len(input_features)})
    dict_feature_category_definitions.update({"feature_name_to_index": dict(zip(input_features,range(len(input_features))))})
    dict_per_feature_category_definitions = {}
    dict_binary_index_to_featcategory_coordinate = {}
    dict_featcategory_coordinate_to_binary_index = {}
    dict_binary_index_to_feature = {}
    dict_feature_to_binary_indexes = {}
    

    size_all_category_space = 0
    for feat_ in input_features:
        categories_ = list(df[feat_].unique())
        no_categories_ = len(categories_)
        dict_feature_to_binary_indexes.update({feat_ : np.asarray(range(size_all_category_space,size_all_category_space+no_categories_)).astype("int")})
        for index_category_ in range(no_categories_):
            binary_index_ = size_all_category_space + index_category_
            dict_binary_index_to_featcategory_coordinate.update({binary_index_ : (feat_, categories_[index_category_])})
            dict_binary_index_to_feature.update({binary_index_ : feat_})
            dict_featcategory_coordinate_to_binary_index.update({(feat_, categories_[index_category_]) : binary_index_})
        size_all_category_space = size_all_category_space + no_categories_
        dict_per_feature_category_definitions.update({feat_ : {"categories": categories_,
                                                              "no_categories": no_categories_,
                                                               "set_categories": set(categories_),
                                                              "category_name_to_index": dict(zip(categories_,range(no_categories_))),
                                                              }})
        
    index_binary_ = 0
    dict_feature_binary_indexes_start_end = {}
    for j in range(dict_feature_category_definitions["no_input_features"]):
        feat_ = input_features[j]
        no_categories_ = dict_per_feature_category_definitions[feat_]["no_categories"]
        start_ind_ = index_binary_
        end_ind_ = index_binary_ + no_categories_
        dict_feature_binary_indexes_start_end.update({feat_ : [start_ind_, end_ind_]})
        index_binary_ = end_ind_
    dict_feature_category_definitions.update({"dict_feature_binary_indexes_start_end":dict_feature_binary_indexes_start_end})

        
    dict_feature_category_definitions.update({"dict_per_feature_category_definitions":dict_per_feature_category_definitions})
    for feat_ in input_features:
        categories_ = dict_per_feature_category_definitions[feat_]
    dict_feature_category_definitions.update({"dict_binary_index_to_featcategory_coordinate":dict_binary_index_to_featcategory_coordinate})
    dict_feature_category_definitions.update({"dict_featcategory_coordinate_to_binary_index":dict_featcategory_coordinate_to_binary_index})
    dict_feature_category_definitions.update({"dict_binary_index_to_feature":dict_binary_index_to_feature})
    dict_feature_category_definitions.update({"dict_feature_to_binary_indexes":dict_feature_to_binary_indexes})
    
    dict_feature_category_definitions.update({"all_indexes":set(df.index)})

    dict_feature_category_definitions.update({"array_binary_index_to_featcategory_coordinate": np.asarray(list(dict_binary_index_to_featcategory_coordinate.values()))})
    dict_feature_category_definitions.update({"array_binary_index_to_feature": np.asarray(list(dict_binary_index_to_feature.values()))})

    dict_feature_category_definitions.update({"size_all_category_space":size_all_category_space})
    base_string_binary_gene = "".join(np.ones(size_all_category_space).astype("int").astype("str"))
    dict_feature_category_definitions.update({"base_string_binary_gene":base_string_binary_gene})
    
    dict_feature_category_definitions.update({"task":task})
    if(task == "classification"):
        target_list_classes = list(df[target_col].unique())
        no_target_list_classes = len(target_list_classes)
        target_category_frequencies = get_category_frequencies(df, target_col, target_list_classes, normalize = True)
    else:
        target_list_classes = None
        no_target_list_classes = None
        target_category_frequencies = None
    
    dict_feature_category_definitions.update({"target_col":target_col})

    dict_feature_category_definitions.update({"target_list_classes":target_list_classes})
    dict_feature_category_definitions.update({"no_target_list_classes":no_target_list_classes})
    dict_feature_category_definitions.update({"target_category_frequencies":target_category_frequencies})

    
    dict_feature_category_definitions.update({"categorical_cols":categorical_cols})
    dict_feature_category_definitions.update({"continous_cols":continous_cols})
    
    dict_feature_category_definitions.update({"no_permutations":no_permutations})

    dict_feature_category_definitions.update({"minimum_confidence_for_a_cohort":minimum_confidence_for_a_cohort})
    dict_feature_category_definitions.update({"minimum_population_for_a_cohort":minimum_population_for_a_cohort})
    dict_feature_category_definitions.update({"maximum_population_for_a_cohort":maximum_population_for_a_cohort})
    dict_feature_category_definitions.update({"no_initial_population_size":no_initial_population_size})
    dict_feature_category_definitions.update({"no_random_additions_per_generation":no_random_additions_per_generation})
    dict_feature_category_definitions.update({"no_mutations_per_generation":no_mutations_per_generation})
    dict_feature_category_definitions.update({"no_crossovers_per_generation":no_crossovers_per_generation})
    dict_feature_category_definitions.update({"top_quantile_to_survive_per_generation":top_quantile_to_survive_per_generation})

    dict_feature_category_definitions.update({"no_generations":no_generations})
    dict_feature_category_definitions.update({"no_bins_discretization":no_bins_discretization})

                                    
    return dict_feature_category_definitions

def binary_gene_to_solution_form(binary_gene, dict_feature_category_definitions):
    binary_indices_valid_ = np.where(np.asarray([*binary_gene])=='1')[0]
    solution_form = dict_feature_category_definitions["array_binary_index_to_featcategory_coordinate"][binary_indices_valid_]
    return solution_form    

def solution_form_to_binary_gene(solution_form, dict_feature_category_definitions):
    dict_featcategory_coordinate_to_binary_index = dict_feature_category_definitions["dict_featcategory_coordinate_to_binary_index"]
    binary_gene = np.zeros(dict_feature_category_definitions["size_all_category_space"]).astype("int").astype("str")
    binary_indices_ = [dict_featcategory_coordinate_to_binary_index[tuple(featcategory_coordinate)] for featcategory_coordinate in solution_form]
    binary_indices_ = np.asarray(binary_indices_).astype("int")
    binary_gene[binary_indices_] = '1'
    binary_gene = "".join(binary_gene)
    return binary_gene
     
def get_index_sets_per_category(df, dict_feature_category_definitions):
    dict_index_sets_per_category = {}
    input_features = dict_feature_category_definitions["input_features"]
    for feat_ in input_features:
        dict_index_sets_per_category.update({feat_ : {}})
        categories_ = dict_feature_category_definitions["dict_per_feature_category_definitions"][feat_]["categories"]
        for category_ in categories_:
            index_set_ = set(df[df[feat_]==category_].index)
            dict_index_sets_per_category[feat_].update({category_ : index_set_})
    return dict_index_sets_per_category



def tuples_to_grouped_lists_(list_of_tuples):
    dict_result = { k : [*map(lambda v: v[1], values)]
        for k, values in groupby(sorted(list_of_tuples, key=lambda x: x[0]), lambda x: x[0])
        }
    return dict_result

def get_index_sets_for_solution_form(solution_form, dict_feature_category_definitions, dict_index_sets_per_category):
    dict_grouped_categories_ = tuples_to_grouped_lists_(solution_form)
    index_set_for_solution = dict_feature_category_definitions["all_indexes"]
    for feat_ in dict_grouped_categories_.keys():
        list_existing_categoies_ = dict_grouped_categories_[feat_]
        index_set_per_feat_ = set()
        for category_ in list_existing_categoies_:
            index_set_per_feat_ = set.union(index_set_per_feat_, dict_index_sets_per_category[feat_][category_])
        index_set_for_solution = set.intersection(index_set_for_solution, index_set_per_feat_)
    return index_set_for_solution

def get_index_sets_for_binary_gene_form(binary_gene, dict_feature_category_definitions, dict_index_sets_per_category):
    solution_form = binary_gene_to_solution_form(binary_gene, dict_feature_category_definitions)
    index_set_for_solution = get_index_sets_for_solution_form(solution_form, dict_feature_category_definitions, dict_index_sets_per_category)
    return index_set_for_solution

def mutate_(solution_form, binary_form, dict_feature_category_definitions):
    dict_grouped_categories_ = tuples_to_grouped_lists_(solution_form)
    mutated_binary_form = list(binary_form)
    ## choose random feature:
    feat_ = random.choice(dict_feature_category_definitions["input_features"])
    set_categories_ = dict_feature_category_definitions["dict_per_feature_category_definitions"][feat_]["set_categories"]
    no_all_categories_ = dict_feature_category_definitions["dict_per_feature_category_definitions"][feat_]["no_categories"]
    current_categories_ = dict_grouped_categories_[feat_]
    current_categories_set_ = set(current_categories_)
    no_current_categories_ = len(current_categories_)
    set_non_existing_categories_ = set_categories_ - current_categories_set_
    if(no_current_categories_ == 1):
        ## can only add a new category
        category_to_add = random.choice(list(set_non_existing_categories_))
        binary_ind_ = dict_feature_category_definitions["dict_featcategory_coordinate_to_binary_index"][(feat_, category_to_add)]
        mutated_binary_form[binary_ind_] = '1'
    elif(no_current_categories_ == no_all_categories_):
        ## can only delete a category
        category_to_delete = random.choice(list(current_categories_set_))
        binary_ind_ = dict_feature_category_definitions["dict_featcategory_coordinate_to_binary_index"][(feat_, category_to_delete)]
        mutated_binary_form[binary_ind_] = '0'
    else:
        ## can flip any of indices 
        category_to_flip = random.choice(dict_feature_category_definitions["dict_per_feature_category_definitions"][feat_]["categories"])
        binary_ind_ = dict_feature_category_definitions["dict_featcategory_coordinate_to_binary_index"][(feat_, category_to_flip)]
        if(mutated_binary_form[binary_ind_] == '0'):
            mutated_binary_form[binary_ind_] = '1'
        else:
            mutated_binary_form[binary_ind_] = '0'
    mutated_binary_form = "".join(mutated_binary_form)
    mutated_solution_form = binary_gene_to_solution_form(mutated_binary_form, dict_feature_category_definitions)
    return mutated_solution_form, mutated_binary_form

def mutate(solution_form, binary_form, dict_feature_category_definitions, set_tried_solutions):
    for j in range(MAX_TRIALS_NEW_AUTHENTIC_SOLUTIONS):
        mutated_solution_form, mutated_binary_form = mutate_(solution_form, binary_form, dict_feature_category_definitions)
        if(not(mutated_binary_form in set_tried_solutions)):
            return mutated_solution_form, mutated_binary_form
    return mutated_solution_form, mutated_binary_form
    
def crossover_(binary_form_1, 
              binary_form_2, 
              dict_feature_category_definitions):
    # for each feature, choose a random parent
    binary_forms_array_ = [np.asarray([*binary_form_1]), np.asarray([*binary_form_2])]
    crossovered_binary_form = np.asarray([])
    dict_feature_binary_indexes_start_end = dict_feature_category_definitions["dict_feature_binary_indexes_start_end"]
    for j in range(dict_feature_category_definitions["no_input_features"]):
        feat_ = dict_feature_category_definitions["input_features"][j]
        start_ind_, end_ind_ = dict_feature_binary_indexes_start_end[feat_]
        ind_parent_ = random.choice([0,1])
        binary_per_feat_ = binary_forms_array_[ind_parent_][start_ind_:end_ind_]
        crossovered_binary_form = np.append(crossovered_binary_form,binary_per_feat_)
    crossovered_binary_form = "".join(crossovered_binary_form)
    crossovered_solution_form = binary_gene_to_solution_form(crossovered_binary_form, dict_feature_category_definitions)
    return crossovered_solution_form, crossovered_binary_form

def crossover(binary_form_1, 
              binary_form_2, 
              dict_feature_category_definitions, set_tried_solutions):
    for j in range(MAX_TRIALS_NEW_AUTHENTIC_SOLUTIONS):
        crossovered_solution_form, crossovered_binary_form =crossover_(binary_form_1, binary_form_2, dict_feature_category_definitions)
        if(not(crossovered_binary_form in set_tried_solutions)):
            return crossovered_solution_form, crossovered_binary_form
    return crossovered_solution_form, crossovered_binary_form

def generate_random_binary_gene(dict_feature_category_definitions):
    for j in range(dict_feature_category_definitions["no_input_features"]):
        feat_ = dict_feature_category_definitions["input_features"][j]
        no_categories_ = dict_feature_category_definitions["dict_per_feature_category_definitions"][feat_]["no_categories"]
        binary_form_ = np.random.randint(0,2, no_categories_)
        ## cant be all zero, choose a random index and flip
        if(binary_form_.sum()==0):
            rand_ind_ = np.random.randint(no_categories_)
            binary_form_[rand_ind_] = 1
        if(j==0):
            binary_form = binary_form_
        else:
            binary_form = np.append(binary_form,binary_form_)
    binary_form = "".join(binary_form.astype("int").astype("str"))
    solution_form = binary_gene_to_solution_form(binary_form, dict_feature_category_definitions)
    return solution_form, binary_form


def generate_random_individual(dict_feature_category_definitions, set_tried_solutions):
    for j in range(MAX_TRIALS_NEW_AUTHENTIC_SOLUTIONS):
        solution_form, binary_form = generate_random_binary_gene(dict_feature_category_definitions)
        if(not(binary_form in set_tried_solutions)):
            return solution_form, binary_form
    return solution_form, binary_form


def get_no_precision_(x):
    try :
        return len(x.split('.')[1])
    except:
        return None
    


def quantile_to_category_(x, no_bins, bin_names_):
    try : 
        bin_ind_ = math.floor((x*no_bins))
        return bin_names_[bin_ind_]
    except:
        return "NaN"
    
def discretize_continous_feature_(df, continous_feature, no_bins):
    len_df = len(df)
    df_binned_ = df[[continous_feature]]
    precision_ = int(df_binned_[continous_feature].astype(str).apply(lambda x : get_no_precision_(x)).dropna().max())


    quantiles_ = (df_binned_.rank() / len_df)

    df_binned_["quantile"] = quantiles_
    quantile_bins_ = [[(j_/no_bins),((j_+1)/no_bins)] for j_ in range(no_bins)]

    bins_limits_ = []
    bin_names_ = []
    for j_ in range(no_bins): 
        bins_limits_.append([df_binned_[continous_feature].quantile(quantile_bins_[j_][0]),df_binned_[continous_feature].quantile(quantile_bins_[j_][1])])
    for j_ in range(no_bins):
        bin_names_.append(str(round(bins_limits_[j_][0],precision_)) + " - " + str(round(bins_limits_[j_][1],precision_)))
    df.loc[:, continous_feature] = df_binned_["quantile"].apply(lambda x : quantile_to_category_(x, no_bins, bin_names_))
    return df

def discretize_continous_features(df, continous_features, no_bins):
    for continous_feature in continous_features:
        df = discretize_continous_feature_(df, continous_feature, no_bins)
    return df


## regression fitness calculation

def calculate_regression_fitness(df, 
                                 target_col, 
                                 no_permutations,
                                 indices_cohort
                                ):
    test_result_ =  scipy.stats.ttest_ind(df.loc[indices_cohort][target_col].values, df[target_col].values , permutations=no_permutations, equal_var=False)
    fitness, confidence = np.abs(test_result_.statistic), 1 - test_result_.pvalue
    return fitness, confidence

############################################################################

## classification fitness calculation
def calculate_classification_fitness(df, 
                                     target_col, 
                                     no_permutations,
                                     target_list_classes, 
                                     target_category_frequencies,
                                     indices_cohort, 
                                    ):
    len_cohort_ = len(indices_cohort)
    if(len_cohort_>0):
        distances_ = []
        distances_control_ = []

        cohort_category_frequencies = get_category_frequencies(df.loc[indices_cohort], target_col, target_list_classes, normalize=True)
        for j in range(no_permutations):
            sample_frequencies_  = get_category_frequencies(df.sample(len_cohort_), target_col, target_list_classes, normalize=True)
            control_target_category_frequencies = get_category_frequencies(df.sample(len_cohort_), target_col, target_list_classes, normalize=False)

            distance_ = jensenshannon(cohort_category_frequencies.values, target_category_frequencies.values)
            distances_.append(distance_)
            distance_control_ =  jensenshannon(cohort_category_frequencies.values, sample_frequencies_.values)
            distances_control_.append(distance_control_)

        distances_ = np.asarray(distances_)   
        distances_control_ = np.asarray(distances_control_)
        distances_diff_ = distances_ - distances_control_
        dof_ = no_permutations
        
        
        t_val_ = distances_diff_.mean() / ((distances_diff_.std()/np.sqrt(no_permutations))) 
        p_val_ = scipy.stats.t.sf(abs(t_val_), dof_)
        confidence = 1 - 2*p_val_
        fitness = abs(t_val_)
        
    
    else : 
        fitness, confidence = 0, 0
    return fitness, confidence
    

def calculate_fitness(df, 
                      target_col, 
                      task,
                      no_permutations,
                      target_list_classes, 
                      target_category_frequencies,
                      indices_cohort, 
                     ):
    if(task == "regression"):
        fitness, confidence = calculate_regression_fitness(df, 
                                 target_col, 
                                 no_permutations,
                                 indices_cohort
                                )
    else:
        fitness, confidence = calculate_classification_fitness(df, 
                                     target_col, 
                                     no_permutations,
                                     target_list_classes, 
                                     target_category_frequencies,
                                     indices_cohort, 
                                    )
    return fitness, confidence

def check_solution_(df,
                    solution_form,
                    binary_form,
                    dict_feature_category_definitions,
                    dict_index_sets_per_category
                ):
    
        target_col = dict_feature_category_definitions["target_col"]

        indices_cohort = get_index_sets_for_solution_form(solution_form, dict_feature_category_definitions, dict_index_sets_per_category)
        df_cohort = df.loc[indices_cohort]
        len_cohort = len(df_cohort)
        solution_valid = None
        fitness = INVALID_COHORT_FITNESS
        confidence = 0.0
        
        minimum_population_for_a_cohort = dict_feature_category_definitions["minimum_population_for_a_cohort"]
        maximum_population_for_a_cohort = dict_feature_category_definitions["maximum_population_for_a_cohort"]
        minimum_confidence_for_a_cohort = dict_feature_category_definitions["minimum_confidence_for_a_cohort"]
        
        if(len_cohort<minimum_population_for_a_cohort):
            solution_valid = False
            fitness = INVALID_COHORT_FITNESS
            return solution_valid, len_cohort, indices_cohort, fitness, confidence
        if(len_cohort>maximum_population_for_a_cohort):
            solution_valid = False
            fitness = INVALID_COHORT_FITNESS
            return solution_valid, len_cohort, indices_cohort, fitness, confidence

        fitness, confidence = calculate_fitness(df, 
                      dict_feature_category_definitions["target_col"], 
                      dict_feature_category_definitions["task"],
                      dict_feature_category_definitions["no_permutations"],
                      dict_feature_category_definitions["target_list_classes"],
                      dict_feature_category_definitions["target_category_frequencies"],                                                   
                      indices_cohort, 
                     )
        if(confidence<minimum_confidence_for_a_cohort):
            solution_valid = False
            fitness = INVALID_COHORT_FITNESS
            return solution_valid, len_cohort, indices_cohort, fitness, confidence
        
        solution_valid = True
        return solution_valid, len_cohort, indices_cohort, fitness, confidence     
    
def get_empty_df_solutions():
    df_solutions = pd.DataFrame([])
    df_solutions['binary_form'] = []
    df_solutions['solution_form'] = []
    df_solutions['solution_valid'] = []
    df_solutions['size_cohort'] = []
    df_solutions['fitness'] = []
    df_solutions['confidence'] = []
    return df_solutions

def prepare_solution_records():
    df_solutions = get_empty_df_solutions()
    set_tried_solutions = set()  ## for binary gene form
    return set_tried_solutions, df_solutions


def add_solution(df,
                 solution_form, binary_form,
                dict_feature_category_definitions,
                dict_index_sets_per_category,
                set_tried_solutions,
                df_solutions,
                ):

    if(not (binary_form in set_tried_solutions)):
        solution_valid, len_cohort, indices_cohort, fitness, confidence  = check_solution_(df,
            solution_form,
            binary_form,
            dict_feature_category_definitions,
            dict_index_sets_per_category
        )
        set_tried_solutions.add(binary_form)
        df_solutions = df_solutions.append([{'binary_form':binary_form,
                                             'solution_form' : solution_form,
                           'solution_valid':bool(solution_valid),
                           'size_cohort':int(len_cohort),
                           'fitness':fitness,
                           'confidence':confidence,
                          }])
    df_solutions.index = range(len(df_solutions))
    return  set_tried_solutions, df_solutions


def add_random_solutions(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions,
                        df_solutions,
                        no_solutions,
                        ):
    
    for j in range(no_solutions):
        solution_form, binary_form = generate_random_individual(dict_feature_category_definitions, set_tried_solutions)
        set_tried_solutions, df_solutions = add_solution(df,
                 solution_form, binary_form,
                dict_feature_category_definitions,
                dict_index_sets_per_category,
                set_tried_solutions,
                df_solutions,
                )
    return  set_tried_solutions, df_solutions

def initate_first_generation(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                            ):
    
    set_tried_solutions, df_solutions = prepare_solution_records()
    no_solutions = dict_feature_category_definitions["no_initial_population_size"]
    set_tried_solutions, df_solutions = add_random_solutions(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions,
                        df_solutions,
                        no_solutions,
                        )
    return set_tried_solutions, df_solutions


def do_random_addition_mutation_and_crossover(df,
                                              dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions, df_solutions,
                                             ):
    
    no_random_additions_per_generation = dict_feature_category_definitions["no_random_additions_per_generation"]
    if(no_random_additions_per_generation>0):
        set_tried_solutions, df_solutions = add_random_solutions(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions,
                        df_solutions,
                        no_random_additions_per_generation,
                        )
    no_mutations_per_generation = dict_feature_category_definitions["no_mutations_per_generation"]
    if(no_mutations_per_generation>0):
        if(len(df_solutions)>0):
            sample_individual = df_solutions.sample()

            sample_individual_solution_form, sample_individual_binary_form = sample_individual["solution_form"].values[0], sample_individual["binary_form"].values[0]
            mutated_individual_solution_form, mutated_individual_binary_form = mutate(sample_individual_solution_form, sample_individual_binary_form, dict_feature_category_definitions, set_tried_solutions)
            set_tried_solutions, df_solutions = add_solution(df,
                     mutated_individual_solution_form, mutated_individual_binary_form,
                    dict_feature_category_definitions,
                    dict_index_sets_per_category,
                    set_tried_solutions,
                    df_solutions,
                    )
        
    no_crossovers_per_generation = dict_feature_category_definitions["no_crossovers_per_generation"]
    if(no_crossovers_per_generation>0):
        if(len(df_solutions)>1):
            sample_individual_1 = df_solutions.sample()
            sample_individual_solution_form_1, sample_individual_binary_form_1 = sample_individual_1["solution_form"].values[0], sample_individual_1["binary_form"].values[0]
            sample_individual_2 = df_solutions.sample()
            sample_individual_solution_form_2, sample_individual_binary_form_2 = sample_individual_2["solution_form"].values[0], sample_individual_2["binary_form"].values[0]
            crossovered_individual_solution_form, crossovered_individual_binary_form = crossover(sample_individual_binary_form_1, sample_individual_binary_form_2, dict_feature_category_definitions, set_tried_solutions)

            set_tried_solutions, df_solutions = add_solution(df,
                    crossovered_individual_solution_form, crossovered_individual_binary_form,
                    dict_feature_category_definitions,
                    dict_index_sets_per_category,
                    set_tried_solutions,
                    df_solutions,
                    )
    return set_tried_solutions, df_solutions


def do_random_addition_mutation_and_crossover(df,
                                              dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions, df_solutions,
                                             ):
    
    no_random_additions_per_generation = dict_feature_category_definitions["no_random_additions_per_generation"]
    if(no_random_additions_per_generation>0):
        set_tried_solutions, df_solutions = add_random_solutions(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions,
                        df_solutions,
                        no_random_additions_per_generation,
                        )
    no_mutations_per_generation = dict_feature_category_definitions["no_mutations_per_generation"]
    if(no_mutations_per_generation>0):
        if(len(df_solutions)>0):
            sample_individual = df_solutions.sample()

            sample_individual_solution_form, sample_individual_binary_form = sample_individual["solution_form"].values[0], sample_individual["binary_form"].values[0]
            mutated_individual_solution_form, mutated_individual_binary_form = mutate(sample_individual_solution_form, sample_individual_binary_form, dict_feature_category_definitions, set_tried_solutions)
            set_tried_solutions, df_solutions = add_solution(df,
                     mutated_individual_solution_form, mutated_individual_binary_form,
                    dict_feature_category_definitions,
                    dict_index_sets_per_category,
                    set_tried_solutions,
                    df_solutions,
                    )
        
    no_crossovers_per_generation = dict_feature_category_definitions["no_crossovers_per_generation"]
    if(no_crossovers_per_generation>0):
        if(len(df_solutions)>1):
            sample_individual_1 = df_solutions.sample()
            sample_individual_solution_form_1, sample_individual_binary_form_1 = sample_individual_1["solution_form"].values[0], sample_individual_1["binary_form"].values[0]
            sample_individual_2 = df_solutions.sample()
            sample_individual_solution_form_2, sample_individual_binary_form_2 = sample_individual_2["solution_form"].values[0], sample_individual_2["binary_form"].values[0]
            crossovered_individual_solution_form, crossovered_individual_binary_form = crossover(sample_individual_binary_form_1, sample_individual_binary_form_2, dict_feature_category_definitions, set_tried_solutions)

            set_tried_solutions, df_solutions = add_solution(df,
                    crossovered_individual_solution_form, crossovered_individual_binary_form,
                    dict_feature_category_definitions,
                    dict_index_sets_per_category,
                    set_tried_solutions,
                    df_solutions,
                    )
    return set_tried_solutions, df_solutions




def do_survival(
                dict_feature_category_definitions, 
                df_solutions):
    df_solutions = df_solutions.sort_values('fitness', ascending=False)
    top_quantile_to_survive_per_generation = dict_feature_category_definitions["top_quantile_to_survive_per_generation"]
    MINIMUM_POPULATION_SIZE_PER_GENERATION = dict_feature_category_definitions["no_initial_population_size"]
    no_current_population = len(df_solutions)
    if(no_current_population>MINIMUM_POPULATION_SIZE_PER_GENERATION):
        no_population_to_survive = int(top_quantile_to_survive_per_generation*no_current_population)
        df_solutions = df_solutions.iloc[:no_population_to_survive]
    df_solutions.index = range(len(df_solutions))
    return df_solutions


def do_generation(
                df,
                dict_feature_category_definitions,
                dict_index_sets_per_category,
                set_tried_solutions, df_solutions):
    
    t_start_ = time.time()
    if(len(df_solutions)>0):
        valid_indices_ = df_solutions.binary_form.drop_duplicates().index
        df_solutions = df_solutions.loc[valid_indices_]
        df_solutions.index = range(len(df_solutions))
        df_cohorts_per_generation = df_solutions[df_solutions.solution_valid==1].dropna()
    else:
        df_cohorts_per_generation = df_solutions.copy()
        
    df_cohorts_per_generation = df_cohorts_per_generation[["binary_form", "solution_form", "fitness", "size_cohort" ,"confidence"]]
    df_cohorts_per_generation.loc[:, "size_cohort"] = df_cohorts_per_generation.loc[:, "size_cohort"].astype('int')
    set_tried_solutions, df_solutions =  do_random_addition_mutation_and_crossover(df,
                                              dict_feature_category_definitions,
                        dict_index_sets_per_category,
                        set_tried_solutions, df_solutions,
                                             )
    df_solutions = do_survival(
                dict_feature_category_definitions, 
                df_solutions)
    
    t_end_ = time.time()
    seconds_generation_ = int(t_end_-t_start_)
    
    dict_generation_stats = {}
    dict_generation_stats.update({'mean_fitness': df_solutions['fitness'].mean()})
    dict_generation_stats.update({'max_fitness': df_solutions['fitness'].max()})
    dict_generation_stats.update({'median_fitness': df_solutions['fitness'].median()})
    dict_generation_stats.update({'valid_solutions_ratio': df_solutions['solution_valid'].mean()})
    dict_generation_stats.update({'seconds_generation': seconds_generation_})

    return set_tried_solutions, df_solutions, df_cohorts_per_generation, dict_generation_stats

def process_cohortization(df_input,
                target_col,
                categorical_cols, 
                continous_cols,
                task,
                minimum_confidence_for_a_cohort, 
                minimum_population_for_a_cohort,
                maximum_population_for_a_cohort,
                no_permutations = 50,
                no_initial_population_size = 250,
                no_generations= 20,
                no_random_additions_per_generation = 100,
                no_mutations_per_generation = 100,
                no_crossovers_per_generation = 100,
                top_quantile_to_survive_per_generation = 0.75,
                no_bins_discretization_continous = 5,
                ):
    df = df_input.copy()
    df.index = range(len(df))
    input_features = []
    input_features.extend(categorical_cols)
    if(len(continous_cols)>0):
        df = discretize_continous_features(df, continous_cols, no_bins_discretization_continous)
        input_features.extend(continous_cols)
    
    dict_feature_category_definitions = get_feature_category_definitions(df, input_features, task, target_col, 
                                     categorical_cols, continous_cols,
                                     minimum_confidence_for_a_cohort, 
                                     minimum_population_for_a_cohort,
                                     maximum_population_for_a_cohort,
                                     no_permutations,
                                    no_initial_population_size,
                                    no_generations,
                                    no_random_additions_per_generation,
                                    no_mutations_per_generation,
                                    no_crossovers_per_generation,
                                    top_quantile_to_survive_per_generation,
                                    no_bins_discretization_continous,
                                    )
    
    
    dict_index_sets_per_category = get_index_sets_per_category(df, dict_feature_category_definitions)
    
    set_tried_solutions, df_solutions = initate_first_generation(df,
                        dict_feature_category_definitions,
                        dict_index_sets_per_category,
                            )
    df_cohorts = []
    df_generations = []
    generation_info_columns = ["generation", "population_size", "mean_fitness", "max_fitness", "median_fitness", "valid_solutions_ratio", "computation_duration_in_seconds"]
    for generation_no in range(no_generations):
        set_tried_solutions, df_solutions, df_cohorts_per_generation, dict_generation_stats = do_generation(
                df,
                dict_feature_category_definitions,
                dict_index_sets_per_category,
                set_tried_solutions, df_solutions)
        df_cohorts_per_generation['generation'] = generation_no
        df_generations.append([generation_no,
                              len(df_solutions),
                              dict_generation_stats['mean_fitness'],
                              dict_generation_stats['max_fitness'],
                              dict_generation_stats['median_fitness'],
                              dict_generation_stats['valid_solutions_ratio'],
                              dict_generation_stats['seconds_generation'],
                              ])
       
        print("generation : {} // population_size : {} // duration (seconds) : {} // mean_fitness : {} // max_fitness : {} //".format(generation_no, len(df_solutions), dict_generation_stats['seconds_generation'], dict_generation_stats['mean_fitness'], dict_generation_stats['max_fitness']))
        df_cohorts.append(df_cohorts_per_generation)
    
    df_cohorts = pd.concat(df_cohorts)
    df_cohorts.index = range(len(df_cohorts))
    df_cohorts = df_cohorts.sort_values('generation', ascending=True)
    valid_indices_ = df_cohorts.binary_form.drop_duplicates().index
    df_cohorts = df_cohorts.loc[valid_indices_]
    df_cohorts = df_cohorts.sort_values('fitness', ascending=False)
    df_cohorts.index = range(len(df_cohorts))
    df_cohorts = df_cohorts.drop('binary_form',axis=1)
    df_generations = pd.DataFrame(data=df_generations, columns=generation_info_columns)
    return df_cohorts, df_generations

def get_cohort(df, solution_form, continous_cols):
    solution_form_list_ = tuples_to_grouped_lists_(solution_form)
    set_cohort_indices = set(df.index)
    current_set_indices_ = set()
    features_ = list(solution_form_list_.keys())
    for j in range(len(features_)):
        feat_ = features_[j]
        current_categories_ = solution_form_list_[feat_]
        current_set_indices_ = set()
        for jj in range(len(current_categories_)):
            value_ = current_categories_[jj]
            if(feat_ in continous_cols):
                if(value_ != 'NaN'):
                    value_ = value_.split(" - ")
                    df_tmp_ = df[(df[feat_]>=float(value_[0])) & (df[feat_]<=float(value_[1]))]
                else:
                    df_tmp_ = df[df[feat_].isna()]
            else:
                df_tmp_ = df[df[feat_]==value_]
            indices_tmp_ = set(df_tmp_.index)
            current_set_indices_ = set.union(current_set_indices_,indices_tmp_)
        set_cohort_indices = set.intersection(set_cohort_indices, current_set_indices_)
        
    df_cohort = df.loc[np.asarray(set_cohort_indices)]
    return df_cohort