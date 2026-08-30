import pandas as pd
import ast  # To safely evaluate the Counter strings
from collections import Counter
import math

# Read the CSV file
file_path = 'result_vqa_full_gpt/vqs_scores.csv'  # Replace with the actual path to your CSV file
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
std_error_0 = {}
for kw in keywords0:
    se_0 = math.sqrt(((p_c0[kw] * (1 - p_c0[kw])) / N_c0) + ((p_m0[kw] * (1 - p_m0[kw])) / N_m0))
    std_error_0[kw] = se_0

keywords1 = n_kw_c1.keys()
std_error_1 = {}
for kw in keywords1:
    se_1 = math.sqrt(((p_c1[kw] * (1 - p_c1[kw])) / N_c1) + ((p_m1[kw] * (1 - p_m1[kw])) / N_m1))
    std_error_1[kw] = se_1

# Lastly calculate the score which is the ratio difference but with a scale according to the std error
vqa_scores_0 = {}
for kw in keywords0:
    vqa_scores_0[kw] = (p_m0[kw] - p_c0[kw]) / std_error_0[kw]

vqa_scores_1 = {}
for kw in keywords1:
    vqa_scores_1[kw] = (p_m1[kw] - p_c1[kw]) / std_error_1[kw]

print("vqa_scores_0", vqa_scores_0)
print("---------------------------------------------------------")
print("vqa_scores_1", vqa_scores_1)
