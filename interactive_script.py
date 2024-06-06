import datetime
import re
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.svm import SVC
import builtins as __builtin__

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
reading_char_factor = 0.02


def fetch_data_from_file(filename):
    with open(filename, 'r') as file:
        data = [line.strip() for line in file.readlines()]
    return data


def validate_email(email):
    pattern = r"(?:[a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
    if re.match(pattern, email, re.IGNORECASE):
        return True
    else:
        return False


def print(message):
    __builtin__.print(message)
    time.sleep(len(message) * reading_char_factor)


if __name__ == '__main__':
    print("--> I am an interactive script, "
          "we will go through the task step by step and show the analysis and results accordingly.")

    print(
        "--> First step: reading 'emails.txt' file and load it to a dataframe for an easier and more efficient "
        "processing.")

    # Reading emails from the file.
    emails = fetch_data_from_file("emails.txt")
    df = pd.DataFrame(emails, columns=['original_email_address'])
    print(f"{len(df)} emails loaded")

    # Filtering the invalid emails.
    print("--> Second step: filtering out the invalid emails.")

    filtered_df = df[df['original_email_address'].apply(validate_email)]
    print(f"--> Filtered emails: {len(filtered_df)}. Dropped emails: {len(df) - len(filtered_df)}")

    df = filtered_df

    # Extracting features.
    print("--> Third step: Features Extraction and assign the label attribute")


    def extract_features(email):
        # domain
        domain_pattern = r'@([a-zA-Z0-9.-]+)\.([a-zA-Z]{2,})'
        domain_match = re.search(domain_pattern, email)

        mail_provider = domain_match.group(1)
        provider_domain_type = domain_match.group(2)

        # username
        username = email.split("@")[0]
        textual_pattern = r'[a-zA-Z]+'
        textual_tokens = re.findall(textual_pattern, username)

        username_length = len(username)
        num_of_textual_tokens = len(textual_tokens)
        first_name = textual_tokens[0].lower()
        first_name_length = len(first_name)
        has_dot = int("." in username)
        has_underscore = int("_" in username)
        has_dash = int("-" in username)
        has_capital_letters = int(bool(re.search(r'[A-Z]', username)))

        # numerical tokens
        numeric_pattern = r'\d+'
        numeric_tokens = re.findall(numeric_pattern, username)

        number_numeric_tokens = len(numeric_tokens)
        possible_age = -1
        numeric_token = -1
        length_of_numeric_token = 0

        if number_numeric_tokens == 1:
            numeric_token = int(numeric_tokens[0])
            length_of_numeric_token = len(numeric_tokens[0])

        # possible_age extraction
        numeric_tokens = sorted(numeric_tokens, key=len, reverse=True)  # to give a priority for the 4-digit number
        # over the 2-digit number if both were meaningful.

        for token in numeric_tokens:
            n_token = int(token)
            if 1924 <= n_token <= 2006:
                possible_age = datetime.datetime.now().year - n_token
                break
            elif len(token) == 2 and 0 <= n_token <= 6:
                possible_age = datetime.datetime.now().year - (2000 + n_token)
                break
            elif 24 <= n_token <= 99:
                possible_age = datetime.datetime.now().year - (1900 + n_token)
                break

        return (
            mail_provider, provider_domain_type, username_length, num_of_textual_tokens, first_name, first_name_length,
            has_dot,
            has_underscore, has_dash, has_capital_letters,
            number_numeric_tokens, numeric_token, length_of_numeric_token, possible_age)


    # Preparing a tabular dataset.
    df[['mail_provider', 'provider_domain_type', 'username_length', 'num_of_textual_tokens', 'first_name',
        'first_name_length', 'has_dot',
        'has_underscore', 'has_dash', 'has_capital_letters',
        'number_numeric_tokens', 'numeric_token', 'length_of_numeric_token', 'possible_age']] = df[
        'original_email_address'].apply(
        lambda email: pd.Series(extract_features(email)))

    bins = [-1, 18, 30, 50, 120]
    labels = ['unsure', 'young', 'medium', 'old']
    df['label'] = pd.cut(df['possible_age'], bins=bins, labels=labels, right=False)

    print("--> The data row now looks like this:")

    print(df.iloc[0].to_json(indent=4))

    # Dataset export for later usage.
    print("--> Exporting dataset to csv file 'processed_dataset.csv'")

    df.to_csv("processed_dataset.csv", index=False)

    # Preprocessing the features.
    print("-->  Encoding, training the model and evaluating under the cross validation setup.")

    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', categories='auto')
    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    encoder = ColumnTransformer(
        transformers=[
            ('one_hot', one_hot_encoder, ["mail_provider", "provider_domain_type"]),
            ('ordinal', ordinal_encoder, ['first_name'])
            # ordinal does not fit with this case. word embedding would fit better, so that similar names should have
            # closer representations.
        ], remainder='passthrough', verbose_feature_names_out=False
    )
    encoder.set_output(transform='pandas')

    # df = df[df['label'] != 'unsure']
    X = df.drop(columns=['original_email_address', 'label'])
    X = encoder.fit_transform(X)
    y = df['label']

    # Training phase
    print("--> Training the model using different ML models")
    exp_models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForestClassifier": RandomForestClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42),
    }

    # Select the best model
    best_result = 0
    best_model = ""
    for name, model in exp_models.items():
        model.fit(X, y)
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        scores_mean = np.mean(scores)
        print(f"{name} accuracy: {scores_mean}")
        if scores_mean > best_result:
            best_result = scores_mean
            best_model = name

    print(
        f"We have chosen the model {best_model} as it performs the best. "
        f"You can try the model yourself.")

    # Testing the model with user data.
    user_input = None
    while True:
        user_input = input("Enter an email address to get results or press q to quit: ")
        if user_input == 'q' or user_input == 'Q': break

        if not validate_email(user_input):
            print("Invalid email address.")
            continue
        else:
            user_input_instance = pd.DataFrame([extract_features(user_input)],
                                               columns=['mail_provider', 'provider_domain_type', 'username_length',
                                                        'num_of_textual_tokens', 'first_name', 'first_name_length',
                                                        'has_dot',
                                                        'has_underscore', 'has_dash', 'has_capital_letters',
                                                        'number_numeric_tokens', 'numeric_token',
                                                        'length_of_numeric_token', 'possible_age'])

            user_input_instance = encoder.transform(user_input_instance)

            class_prediction = exp_models[best_model].predict(user_input_instance)
            class_probabilities = exp_models[best_model].predict_proba(user_input_instance)

            final_result = {"age": class_prediction[0], "score": class_probabilities[0][
                exp_models[best_model].classes_.tolist().index(class_prediction[0])]}

            print(final_result)
