import pandas as pd
import ast  # To safely evaluate the Counter strings
from collections import Counter
import math
from tabulate import tabulate
import os

# Read the CSV file
classname = ["landbird", "waterbird"]
run = "full_gpt"  # ["full_gpt", "baseline", "multicap"]
file_path = f'result_vqa_{run}/vqa_scores.csv'  # Replace with the actual path to your CSV file
save_results = True
diff_path_0 = f'result_vqa_{run}/vqa_scores_tab_{classname[0]}.csv'
diff_path_1 = f'result_vqa_{run}/vqa_scores_tab_{classname[1]}.csv'


def calculate_standard_error(keywords, p_c, p_m, N_c, N_m):
    std_error = {}
    for kw in keywords:
        se = math.sqrt(((p_c[kw] * (1 - p_c[kw])) / N_c) + ((p_m[kw] * (1 - p_m[kw])) / N_m))
        if abs(se) <= 0.000001:
            se = 0.000001
        std_error[kw] = se
    return std_error


def calculate_vqa_scores(keywords, p_c, p_m, std_error):
    vqa_scores = {}
    for kw in keywords:
        vqa_scores[kw] = (p_m[kw] - p_c[kw]) / std_error[kw]
    return vqa_scores


def print_vqa_score(classname, vqa_scores, p_c, p_m, std_error):
    result = {
            "Keyword": [],
            "Score": [],
            "Ratio cor.": [],
            "Ratio wrong.": [],
            "SE": [],
            "Acc.": []
        }

    for kw in vqa_scores.keys():
        result["Keyword"].append(kw)
        result["Score"].append(vqa_scores[kw])
        result["Ratio cor."].append(p_c[kw])
        result["Ratio wrong."].append(p_m[kw])
        result["SE"].append(std_error[kw])
        result["Acc."].append(0)

    diff = pd.DataFrame(result)
    diff = diff.sort_values(by=["Score"], ascending=False)
    print(f"VQA scores of class {classname}:")
    print(tabulate(diff, headers='keys', showindex=False))
    return diff


if __name__ == "__main__":
    df = pd.read_csv(file_path)

    # Get the value from 'class_0' where 'metric' is 'correct_keyword_occurrences'
    str_keyword_occur_class0_correct = df.loc[df['Metric'] == 'correct_keyword_occurrences', 'Class_0'].iloc[0]
    str_keyword_occur_class0_wrong = df.loc[df['Metric'] == 'wrong_keyword_occurrences', 'Class_0'].iloc[0]
    str_keyword_occur_class1_correct = df.loc[df['Metric'] == 'correct_keyword_occurrences', 'Class_1'].iloc[0]
    str_keyword_occur_class1_wrong = df.loc[df['Metric'] == 'wrong_keyword_occurrences', 'Class_1'].iloc[0]

    # Remove the "Counter(...)" wrapper and evaluate the dictionary inside
    # Total number of how often was keyword of class 0 and 1 found in correctly (n_kw_c0, n_kw_c1) and incorrectly (n_kw_m0, n_kw_m1) classified images
    n_kw_c0 = Counter(ast.literal_eval(str_keyword_occur_class0_correct[len("Counter("):-1]))
    n_kw_m0 = Counter(ast.literal_eval(str_keyword_occur_class0_wrong[len("Counter("):-1]))
    n_kw_c1 = Counter(ast.literal_eval(str_keyword_occur_class1_correct[len("Counter("):-1]))
    n_kw_m1 = Counter(ast.literal_eval(str_keyword_occur_class1_wrong[len("Counter("):-1]))

    # total images of class 0 and 1 that were correctly (N_c0 and N_c1) and incorrectly (N_m0, N_m1) classified
    N_c0 = int(df.loc[df['Metric'] == 'total_correct', 'Class_0'].iloc[0])
    N_m0 = int(df.loc[df['Metric'] == 'total_wrong', 'Class_0'].iloc[0])
    N_c1 = int(df.loc[df['Metric'] == 'total_correct', 'Class_1'].iloc[0])
    N_m1 = int(df.loc[df['Metric'] == 'total_wrong', 'Class_1'].iloc[0])

    # Divide how often the keywords were found in all correct/wrong classified images
    # Ratio of how often was keyword of class 0 and 1 found in correctly (p_c0, p_c1) and incorrectly (p_m0, p_m1) classified images
    p_c0 = Counter({key: value / N_c0 for key, value in n_kw_c0.items()})
    p_m0 = Counter({key: value / N_m0 for key, value in n_kw_m0.items()})
    p_c1 = Counter({key: value / N_c1 for key, value in n_kw_c1.items()})
    p_m1 = Counter({key: value / N_m1 for key, value in n_kw_m1.items()})

    # Calculate the Standard Error (SE)
    keywords0 = n_kw_c0.keys()
    keywords1 = n_kw_c1.keys()
    std_error_0 = calculate_standard_error(keywords0, p_c0, p_m0, N_c0, N_m0)
    std_error_1 = calculate_standard_error(keywords1, p_c1, p_m1, N_c1, N_m1)

    # Lastly calculate the score which is the ratio difference but with a scale according to the std error
    vqa_scores_0 = calculate_vqa_scores(keywords0, p_c0, p_m0, std_error_0)
    vqa_scores_1 = calculate_vqa_scores(keywords1, p_c1, p_m1, std_error_1)

    # Finally print and optionally safe the results
    diff_0 = print_vqa_score(classname[0], vqa_scores_0, p_c0, p_m0, std_error_0)
    diff_1 = print_vqa_score(classname[1], vqa_scores_1, p_c1, p_m1, std_error_1)

    if save_results:
        if os.path.exists(diff_path_0) or os.path.exists(diff_path_1):
            print("Files already exist! Files are NOT overwritten!")
            choice = input("Do you want to overwrite the file [Y,N]:")
            if choice == "Y":
                diff_0.to_csv(diff_path_0)
                diff_1.to_csv(diff_path_1)
