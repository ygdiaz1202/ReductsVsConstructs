import pandas as pd
import numpy as np
import arff
import scipy.io.arff as arff_io
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import warnings
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from os.path import isfile, join
import multiprocessing
import os
import subprocess
import time
import shutil
from scipy.stats import mode
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from scipy import stats
import json
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import wilcoxon


def add_noise_to_arff(input_arff_filename, output_path, categorical_noise_level=0.01):
    # Load ARFF dataset
    with open(input_arff_filename, 'r') as arff_file:
        arff_data = arff.load(arff_file)
    data, meta = arff_data['data'], arff_data['attributes']
    df = pd.DataFrame(data)

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Add noise to numeric columns
    # numeric_columns = df.select_dtypes(include=['float64']).columns
    # for col in numeric_columns:
    #     if col != arff_data['attributes'][-1][0]:  # Exclude class attribute from encoding
    #         df[col] += np.random.normal(loc=0, scale=numeric_noise_level, size=len(df))

    # Add noise to categorical columns
    all_columns = df.columns
    for col in all_columns:
        mask = np.random.rand(len(df)) < categorical_noise_level
        categories = df[col].unique()
        df.loc[mask, col] = np.random.choice(categories, sum(mask))

    # Save the noisy data back to ARFF format
    noisy_data = df.to_records(index=False)
    arff_data = np.array(noisy_data.tolist(), dtype=object)

    new_filename = str(os.path.basename(input_arff_filename).replace('_train', '_with_noise_train'))
    new_filepath = os.path.join(output_path, new_filename)

    arff_dict = {'relation': new_filename.replace('.arff', ''), 'attributes': meta, 'data': arff_data}
    with open(new_filepath, 'w') as new_arff_file:
        arff.dump(arff_dict, new_arff_file)


def load_arff_to_dataframe(arff_file):
    with open(arff_file, 'r') as f:
        arff_data = arff.load(f)
    # Check if 'attributes' key exists and is a list
    if 'attributes' in arff_data and isinstance(arff_data['attributes'], list):
        # Convert ARFF data to a Pandas DataFrame
        df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])

        # Check if the last attribute is nominal
        if len(arff_data['attributes']) > 0 and 'nominal' in str(arff_data['attributes'][-1][1]).lower():
            df.iloc[:, -1] = df.iloc[:, -1].astype(str)
        # # Assuming df is your DataFrame with mixed data types including categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns
        encoding_mapping = {}  # Store encoding mapping for each categorical column
        for column in categorical_columns:
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
            encoding_mapping[column] = {label: index for index, label in enumerate(label_encoder.classes_)}
            # Update attribute information in the ARFF header with encoded values
        attributes_encoded = []
        for attr_name, attr_type in arff_data['attributes']:
            if attr_name in encoding_mapping:
                attributes_encoded.append((attr_name, 'NUMERIC'))
            else:
                attributes_encoded.append((attr_name, attr_type))

        # Assuming X is your feature matrix with missing values
        # Initialize the SimpleImputer
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')  #You can choose other strategies like 'median', 'most_frequent', etc.

        # Fit the imputer to your data
        imputer.fit(df)

        # Transform your data, replacing missing values
        df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)

        # Split the DataFrame into training and testing sets
        data_train, data_test = train_test_split(df_imputed, test_size=0.10, random_state=45)
        return [data_train, data_test, attributes_encoded]
    else:
        raise ValueError("Invalid ARFF file format.")


def read_arff_dataset(file_path):
    # Read ARFF file
    data, meta = arff_io.loadarff(file_path)
    # Convert data to a Pandas DataFrame
    df = pd.DataFrame(data)
    return df


def read_train_files(_dir_, _extension_):
    return [f for f in os.listdir(_dir_) if isfile(join(_dir_, f)) and '_train' in f and os.path.splitext(f)[1] == _extension_]


def read_arff_files(_dir_):
    return [f for f in os.listdir(_dir_) if isfile(join(_dir_, f)) and os.path.splitext(f)[1] == '.arff']


def read_txt_files(_dir_):
    return [f for f in os.listdir(_dir_) if isfile(join(_dir_, f)) and os.path.splitext(f)[1] == '.txt']


def delete_file(file):
    if os.path.exists(file):
        os.remove(file)
        return True
    return False


def train_test_split_to_arff(_arff_directory_, arff_filename, dest_dir):
    # Get the filename without extension
    base_name = os.path.splitext(arff_filename)[0]
    print(base_name)
    # Construct paths for saving the training and testing ARFF files
    save_train_path = os.path.join(dest_dir, base_name + '_train.arff')
    save_test_path = os.path.join(dest_dir, base_name + '_test.arff')
    # load the dateset and split it into training and testing sets
    full_path = os.path.join(_arff_directory_, arff_filename)
    data_train, data_test, attributes = load_arff_to_dataframe(full_path)
    # Save the training set as ARFF
    with open(save_train_path, 'w') as f_train:
        arff.dump({'relation': base_name + '_train', 'attributes': attributes, 'data': data_train.values.tolist()},
                  f_train)
    # Save the testing set as ARFF
    with open(save_test_path, 'w') as f_test:
        arff.dump({'relation': base_name + '_test', 'attributes': attributes, 'data': data_test.values.tolist()},
                  f_test)
    return True


def split_all_arff_and_save(_arff_directory_, dest_dir):
    arff_files = read_arff_files(_arff_directory_)
    if not os.path.exists(dest_dir):
        # Create the directory if it does not exist
        print(f'The directory \'{dest_dir}\' does not exist, creating the directory...')
        os.makedirs(dest_dir)

    for arff_file_name in arff_files:
        train_test_split_to_arff(_arff_directory_, arff_file_name, dest_dir)


def execute_jar_file(*args):
    # the java program
    s = subprocess.check_output(['java', '-Xmx5120m', '-XX:-UseGCOverheadLimit', '-jar']+list(args), shell=False)
    text = s.decode("utf-8")
    text = text.replace("\r", "")
    return text


def execute_algorithm(parameters):
    algorithm, file_name = parameters
    alg_name = os.path.splitext(algorithm)[0] # file name without extension
    args = [algorithm, file_name]
    output = execute_jar_file(*args)
    with open(alg_name+"_logFile.log", "a+") as outfile:
        outfile.write(output+'\n')


def execute_algorithm_multi_process(src, algorithms, src_dir, src_extension, dest_dir, dest_extension, num_workers):
    src_files = read_train_files(src_dir, src_extension)
    copied =[]
    for i in range(len(algorithms)):
        if not os.path.isfile(os.path.join(src_dir, algorithms[i])):
            shutil.copy(os.path.join(src, algorithms[i]), src_dir)
            copied.append(True)
        else:
            copied.append(False)
    start_time = time.time()
    current_dir = os.getcwd()
    os.chdir(src_dir)
    # filling the parameters list with the algorithms and files to process
    parameters = []
    for file in src_files:
        for algorithm in algorithms:
            parameters.append((algorithm, file))

    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map_async(execute_algorithm, parameters)
        results.wait()
    end_time = time.time()
    os.chdir(current_dir)

    for i in range(len(algorithms)):
        if copied[i]:
            delete_file(os.path.join(src_dir, algorithms[i]))
    if src_dir != dest_dir:
        dest_files = read_train_files(src_dir, dest_extension)
        for dest_f in dest_files:
            full_path = os.path.join(src_dir, dest_f)
            shutil.copy(full_path, dest_dir)
            delete_file(full_path)
    print("Time for MultiProcessingSquirrel: %ssecs" % (end_time - start_time))


def compute_binary_matrices(arff_dir='./resources/train_test_datasets/',
                            binary_matrices_dir='./resources/binary_matrices/',
                            num_workers=10):
    src = './src/'
    arff_to_binary_m = ['BinaryMatrix.jar']
    print('Computing binary matrices...')
    if not os.path.exists(binary_matrices_dir):
        # Create the directory if it does not exist
        print(f'The directory \'{binary_matrices_dir}\' does not exist, creating the directory...')
        os.makedirs(binary_matrices_dir)
    execute_algorithm_multi_process(src, arff_to_binary_m, arff_dir, '.arff',
                                    binary_matrices_dir, '.bol', num_workers)
    print('Done!')


def compute_reducts_and_constructs(binary_matrices_dir='./resources/binary_matrices/',
                                   const_reducts_dir='./resources/shortest_red_constr/',
                                   num_workers=10):
    src = './src/'
    arff_to_binary_m = ['RCC-MAS.jar']
    print('Computing Reducts and Constructs...')
    if not os.path.exists(const_reducts_dir):
        # Create the directory if it does not exist
        print(f'The directory \'{const_reducts_dir}\' does not exist, creating the directory...')
        os.makedirs(const_reducts_dir)
    execute_algorithm_multi_process(src, arff_to_binary_m,
                                    binary_matrices_dir,
                                    '.bol',
                                    const_reducts_dir,
                                    '.txt',
                                    num_workers)
    print('Done!')


def keep_only_shortest(const_reducts_dir='./resources/shortest_red_constr/',
                       all_constr_red_dir='./resources/all_reducts_and_constructs/'):
    # this function move the files containing all the reducts or constructs to ./resources/all_constr_red_dir/'
    # keeping only in the src directory the files containing the shortest reducts or constructs
    if const_reducts_dir != all_constr_red_dir:
        if not os.path.exists(all_constr_red_dir):
            # Create the directory if it does not exist
            print(f'The directory \'{all_constr_red_dir}\' does not exist, creating the directory...')
            os.makedirs(all_constr_red_dir)
        dest_files = read_train_files(const_reducts_dir, '.txt')
        print(f'Moving the files containing all the reducts and constructs to \'{all_constr_red_dir}\'...')
        for all_constr_red in dest_files:
            if 'shortest' not in all_constr_red:
                shutil.copy(os.path.join(const_reducts_dir, all_constr_red), all_constr_red_dir)
                delete_file(os.path.join(const_reducts_dir, all_constr_red))
        print('finished moving the files containing all the reducts and constructs')


def map_attr_subsets_to_dataset(_base_dir_):
    dic_attr_subsets = {}
    _files_list_ = read_txt_files(_base_dir_)

    for _file in _files_list_:
        file_path = os.path.join(_base_dir_, _file)
        with open(file_path) as fp:
            # all the lines in the files must follow the
            # format: [1,2,3,4,5,6] for a reduct/construct {c1,c2,c3,c4,c5,c6} except the first two lines
            _tmp_list_ = [
                [int(i) for i in line.replace(' ', '').replace('[', '').replace(']', '').split(',')]
                for line in fp.readlines()[2:]
            ]
        new_filename = _file.split('_train')[0]  # the train datasets should contain _train
        if new_filename not in dic_attr_subsets:
            dic_attr_subsets[new_filename] = [
                [],  # List of shortest reducts
                []   # List of shortest constructs
            ]
        if 'IndiscM' in _file:
            # if the file contains IndiscM is the file containing the constructs
            dic_attr_subsets[new_filename][1] = _tmp_list_
        else:
            # in other case the file contains reducts
            dic_attr_subsets[new_filename][0] = _tmp_list_
    return dic_attr_subsets


def predict_class(x_train, x_train_labels, y_test):
    # here you can use the model you prefer as base classifier
    _classifier_ = HistGradientBoostingClassifier()
    # _classifier_ = RandomForestClassifier()
    # _classifier_ = DecisionTreeClassifier()
    # _classifier_ = SVC(kernel="rbf", gamma=0.5, C=1.0)
    _classifier_.fit(x_train, x_train_labels)
    _y_pred_ = _classifier_.predict(y_test)
    return [_y_pred_, _classifier_]


def train_model(x_train, x_test, y_train, y_test, list_features):
    list_pred = []
    _stack_ = []
    # att_subset_selected=[]
    n_classifier = (len(list_features))

    # we are going to train a classifier for each readuct/construct and stack them to create the final classifier
    for k in range(n_classifier):
        _train_model_ = x_train[:, list_features[k]]
        validation_model = x_test[:, list_features[k]]

        class_predicted, ck = predict_class(_train_model_, y_train, validation_model)
        _stack_.append(ck)  # Se adiciona al stack el clasificador entrenado
        list_pred.append(class_predicted)  # y la clase que se predice con ese clasificador

    # Combine predictions using a simple voting mechanism (you can use other strategies)
    show_classification_report(list_pred, y_test)
    return _stack_


def show_classification_report(list_pred, y_test):
    ensemble_pred, _ = mode(list_pred, axis=0, keepdims=True)
    ensemble_pred = ensemble_pred.T
    _accuracy_ = accuracy_score(y_test, ensemble_pred)
    _cm_ = confusion_matrix(y_test, ensemble_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _cr_ = classification_report(y_test, ensemble_pred, zero_division=1)
    print("Showing the confusion matrix")
    print(_cm_)
    print("Showing the classification report")
    print(_cr_)
    print("Showing accuracy")
    print(_accuracy_)
    return _accuracy_


def test_model(x_test, y_test, _stack_, list_selected_features):
    counter = 0
    predicted_class = []

    for features in list_selected_features:
        x_test_new = x_test[:, features]
        predicted_class.append(_stack_[counter].predict(x_test_new))
        counter = counter + 1
    return show_classification_report(predicted_class, y_test)


def run_model(_base_dir_, _file_name_, _dict_info_=None, reducts_constr_dir='./resources/shortest_red_constr/'):

    attr_subsets_dict = map_attr_subsets_to_dataset(reducts_constr_dir)

    if _dict_info_ is None:
        _dict_info_ = {}  # Create a new dictionary if not provided
    info_lis = []
    train_file_name = os.path.join(_base_dir_, _file_name_ + "_train.arff")
    df_train = read_arff_dataset(train_file_name)

    test_file_name = os.path.join(_base_dir_, _file_name_.replace('_with_noise', '') + "_test.arff")

    df_test = read_arff_dataset(test_file_name)
    print('Size of the train: {}, size of the test: {}'.format(df_train.shape, df_test.shape))

    info_lis.append(_file_name_)
    print("Dataset name: ", _file_name_)
    print('Training the model using the shortest reducts...')
    shortest_red = attr_subsets_dict[_file_name_][0]
    info_lis.append(len(shortest_red))
    info_lis.append(len(shortest_red[0]))

    # Separate features and target variable
    x = df_train.iloc[:, :-1].values  # Exclude the last column
    y = df_train.iloc[:, -1].values  # Take only the last column
    # split the training set in train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

    # this is the test dataset don't should be used in the training process
    x_test = df_test.iloc[:, :-1].values
    y_test = df_test.iloc[:, -1].values

    ensemble_red = train_model(x_train, x_test, y_train, y_test, shortest_red)
    print('Training done!')
    print('Testing the model for reducts...')
    reducts_accuracy = test_model(x_test, y_test, ensemble_red, shortest_red)
    info_lis.append(reducts_accuracy)
    print('Testing done!')

    print('Training the model using the shortest constructs...')
    shortest_constr = attr_subsets_dict[_file_name_][1]
    info_lis.append(len(shortest_constr))
    info_lis.append(len(shortest_constr[0]))
    ensemble_constr = train_model(x_train, x_test, y_train, y_test, shortest_constr)
    print('Training done!')
    print('Testing the model for constructs...')
    constructs_accuracy = test_model(x_test, y_test, ensemble_constr, shortest_constr)
    info_lis.append(constructs_accuracy)
    print('Testing done!')
    _dict_info_[_file_name_] = info_lis
    return _dict_info_


def create_csv_file(_results_dict_, _file_name_='results.csv'):
    files_info = [["Name", "Number of reducts", "Shortest reducts Size", 'Reducts accuracy',
                   "Number of constructs", "Shortest constructs Size", 'Constructs accuracy']]
    for key in _results_dict_.keys():
        files_info.append(_results_dict_[key])
    with open(_file_name_, 'w', newline='') as f:
        writer = csv.writer(f)
        # Writing each row of the matrix to the CSV file
        for row in files_info:
            writer.writerow(row)


def read_dict_from_csv(csv_file):
    data_dict = {}
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data_dict[row[0]] = row
    return data_dict


def run_report_for_all_files(_base_dir_, report_file_name='results.csv', reducts_constr_dir= './resources/shortest_red_constr/'):
    train_files_name = read_train_files(_base_dir_, '.arff')
    _results_dict_ = {}
    for file_name in train_files_name:
        _file_name_ = file_name.split('_train')[0]  # the files should be named as 'base_name_train_other.arff'
        try:
            run_model(_base_dir_, _file_name_, _results_dict_, reducts_constr_dir)
        except Exception as e:
            print(f"An error occurred in the processing of {_file_name_}: {e}")
    create_csv_file(_results_dict_, report_file_name)
    return _results_dict_


def add_noise_to_all_files(_base_dir_, output_arff_path, nose_level=0.1):
    arff_train_files = read_train_files(_base_dir_, '.arff')
    for file_name in arff_train_files:
        print(file_name)
        # we added some noise to the train dataset
        add_noise_to_arff(os.path.join(_base_dir_, file_name), output_arff_path, nose_level)
        # the original test dataset should be copied without adding any noise
        shutil.copy(os.path.join(_base_dir_, file_name.replace('_train', '_test')), output_arff_path)


def compute_reducts_and_constructs_noise_example(noise_level, base_dir='./resources/arff_with_noise/'):
    binary_m_dir = os.path.join(base_dir, 'bm_'+str(noise_level))
    compute_binary_matrices(base_dir, binary_m_dir)
    shortest_dir = os.path.join(base_dir, 'shortest_noise_'+str(noise_level))
    compute_reducts_and_constructs(binary_m_dir, shortest_dir)
    keep_only_shortest(shortest_dir, os.path.join(base_dir, 'all_'+str(noise_level)))


def run_example_noise_data(base_dir='./resources/train_test_datasets/'):
    for noise_level in [0.1, 0.2, 0.3]:  # change this to variate the level of noise
        noise_datasets_path = './resources/arff_with_noise'+str(noise_level)+'/'
        # add_noise_to_all_files(base_dir, noise_datasets_path)
        shortest_attr_subset_dir = os.path.join(noise_datasets_path, 'shortest_noise_'+str(noise_level))
        # compute_reducts_and_constructs_noise_example(noise_level, noise_datasets_path)
        dict_info = run_report_for_all_files(noise_datasets_path, 'results_noise_'+str(noise_level/10)+'.csv', shortest_attr_subset_dir)
        save_dict(dict_info, 'results_noise_'+str(noise_level/10)+'.json')
        p_value = prob_test(dict_info)
        with open("statist_"+str(noise_level/10)+'.txt', "w") as file:
            # Write the string to the file
            file.write('p_value = '+str(p_value))


def run_example_single_dataset():
    base_dir = './resources/train_test_datasets/'
    reduct_constr_dir = './resources/shortest_red_constr/'
    base_file_name = 'Mushroom-no-missing-values'
    results_dict = {}
    run_model(base_dir, base_file_name, results_dict, reduct_constr_dir)
    print(results_dict)


def get_decision_rules(df_train, _attr_subsets_):
    # Define an empty list to store rules
    rules = []
    # Separate features and target variable
    x_train = df_train.iloc[:, :-1].values  # Exclude the last column
    y_train = df_train.iloc[:, -1].values  # Take only the last column

    # split the training set in train and test
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=validate_percent, random_state=42)
    # Generate decision tree rules for each attr subset
    for attr_subset in _attr_subsets_:
        # Train decision tree classifier using selected subset of attributes
        classifier = DecisionTreeClassifier()
        classifier.fit(x_train[:, attr_subset], y_train)

        # Export decision tree rules as text
        rules_text = export_text(classifier, feature_names=df_train.columns[attr_subset].tolist())
        rules.append(rules_text)
        # Print rules_text to inspect the format
        print("Decision tree rules:")
    # Combine rules
    combined_rules = '\n'.join(rules)
    return combined_rules


# Define functions to parse rules and classify new instances
def parse_rule(rule):
    lines = rule.split('\n')
    attribute = None
    threshold = None
    class_label = None
    for line in lines:
        if '|---' in line:
            parts = line.split()
            attribute = parts[1]  # Extract attribute name
            threshold = float(parts[2])  # Extract threshold value
        elif '|   |--- class:' in line:
            class_label = float(line.split(':')[-1].strip())  # Extract class label
    return attribute, threshold, class_label


def classify_instance(rule, instance):
    attribute, threshold, class_label = parse_rule(rule)
    while class_label is None:
        if instance[int(attribute.split('_')[-1])] <= threshold:
            rule = rule[rule.find('  |---') + 6:]
        else:
            rule = rule[rule.find('  |   |---') + 10:]
        attribute, threshold, class_label = parse_rule(rule)
    return class_label


def generate_decision_tree(df_train, df_test, _attr_subsets_):
    rules = get_decision_rules(df_train, _attr_subsets_)
    # Initialize list to store predicted labels for test set
    y_pred_test = []

    # Apply rules to each instance in the test set
    for _, instance in df_test.iterrows():
        y_pred_test.append(classify_instance(rules, instance))
    # Evaluate accuracy
    y_test = df_test.iloc[:, -1].values
    accuracy_combined = accuracy_score(y_test, y_pred_test)
    print("\nAccuracy using combined rules:", accuracy_combined)


def test_decision_tree():
    # In test: doesn't work well
    _dir_ = './resources/train_test_datasets/'
    attr_subsets_dict = map_attr_subsets_to_dataset('./resources/shortest_red_constr/')
    df_test = read_arff_dataset(_dir_+'loan_test.arff')
    df_train = read_arff_dataset(_dir_+'loan_train.arff')
    attribute_subsets = attr_subsets_dict['loan'][1]
    print(attribute_subsets)
    generate_decision_tree(df_train, df_test, attribute_subsets)


def prob_test(dict_info):
    sample1_reducts = []
    sample2_constructs = []
    for key in dict_info.keys():
        sample1_reducts.append(float(dict_info[key][3]))
        sample2_constructs.append(float(dict_info[key][6]))
    t_statistic, p_value = stats.ttest_ind(sample2_constructs, sample1_reducts)

    print("T-statistic:", t_statistic)
    print("P-value:", p_value)
    return p_value


def save_dict(my_dict, name):
    # Save dictionary to a file
    with open(name, 'w') as f:
        json.dump(my_dict, f)


def read_dict(dict_name):
    with open(dict_name, 'r') as f:
        loaded_dict = json.load(f)
    return loaded_dict


if __name__ == '__main__':
    # arff_directory = './resources/arff/'
    # split_all_arff_and_save(arff_directory, './resources/train_test_datasets/')
    # compute_binary_matrices()
    # compute_reducts_and_constructs()
    # keep_only_shortest()
    #
    base_dir = './resources/train_test_datasets/'
    # base_file_name = 'Dermatology-no-missing-values'
    # results_dict = {}
    # run_model(base_dir, base_file_name, results_dict)
    # dict_info = run_report_for_all_files(base_dir)

    # dict_info = read_dict_from_csv('./HistGradientBoostingClassifier/results_noise_0.03.csv')
    # p_value = prob_test(dict_info)
    # with open("statist_" + str(0.03 / 10) + '.txt', "w") as file:
    #     # Write the string to the file
    #     file.write('p_value = ' + str(p_value))

    
    # run_example_noise_data()

    # run_example_single_dataset()

    # test_decision_tree()
