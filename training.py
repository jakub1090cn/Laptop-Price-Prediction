import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
import joblib


def read_and_preprocess_data(source):
    """
    Reads data from a CSV file, preprocesses it by applying various cleaning steps,
    and returns the cleaned DataFrame.

    The preprocessing steps include:
    - Invoking a preprocessing function that may apply encoding, normalization, etc.
    - Dropping rows with any missing values. I am performing a drop of missing values to remove rows that contain
    incomplete information about the subject.
    - Removing duplicate rows, keeping the first occurrence.
    - Removing outliers based on certain criteria defined in a separate function.

    Parameters:
    - source (str): The path to the CSV file containing the dataset to be processed.

    Returns:
    - pd.DataFrame: A DataFrame containing the cleaned and preprocessed data.

    Note:
    - The function assumes that the CSV file is encoded in 'latin-1'.
    - It's crucial that the 'preprocess_data' and 'remove_outliers' functions are defined
      and correctly implement the necessary preprocessing and outlier removal steps.
    """
    df = pd.read_csv(source, encoding='latin-1')
    df = preprocess_data(df)
    df.dropna(inplace=True)
    df.drop_duplicates(keep='first', inplace=True)
    df = remove_outliers(df)
    return df


def convert_to_gb(value):
    """
    Converts memory size from TB/GB to GB.

    Parameters:
    - value (str): The memory size as a string with units (e.g., '512GB', '1TB').

    Returns:
    - float: The memory size converted to gigabytes (GB). Returns None if the pattern does not match.
    """
    unit_map = {'TB': 1024, 'GB': 1}
    match = re.search(r'(\d+)(TB|GB)', value, re.IGNORECASE)
    if match:
        num, unit = match.groups()
        return float(num) * unit_map[unit.upper()]
    else:
        return None


def trim_processor(value):
    """
    Trims the processor name to a standardized format.

    For Intel processors, retains the first three words. For others, keeps the first two words.

    Parameters:
    - value (str): The full processor name.

    Returns:
    - str: The trimmed processor name.
    """
    if value.startswith('Intel'):
        return ' '.join(value.split(' ', 3)[:3])
    else:
        return ' '.join(value.split(' ', 2)[:2])

def preprocess_data(df):
    """
    Applies various preprocessing steps to the DataFrame to clean and prepare the data for modeling.

    Steps include:
    - Categorizing rare company names under 'Others'.
    - Standardizing Operating System names.
    - Extracting and converting memory size to GB.
    - Splitting the CPU column into processor name and clock speed, then standardizing processor names.
    - Removing units from numerical columns (e.g., 'GHz', 'GB', 'kg').
    - Simplifying GPU names to brands.
    - Extracting screen width from the screen resolution.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with laptop specifications.

    Returns:
    - pd.DataFrame: The processed DataFrame with standardized and numerical-only features.
    """

    # When analyzing data, we encounter a few rare companies. Training a model with such a limited dataset could be
    # impossible,so I decided to create a category 'Others' that encompasses all these rare companies.
    brand_counts = df['Company'].value_counts()
    small_brands = brand_counts[brand_counts < 10].index
    df['Company'] = df['Company'].apply(lambda x: "Others" if x in small_brands else x)

    # In the dataset, there are a few Operating Systems that can be categorized as more common OSes. To streamline this,
    # I decided to unify certain Operating Systems by replacing specific versions with their more general counterparts.
    df['OpSys'] = df['OpSys'].replace({'Windows 10 S': 'Windows 10', 'Mac OS X': 'macOS'})

    # In the dataset, some Operating Systems are less common. To improve predictions, I categorized these rare Operating
    # Systems under a unified label named 'OtherSys'. This was accomplished by replacing occurrences of 'Chrome OS' and
    # 'Android' in the 'OpSys' column of the dataframe df with 'OtherSys'.
    df['OpSys'] = df['OpSys'].replace(['Chrome OS', 'Android'], 'OtherSys')

    # The 'Memory' column contains both the value and type of memory. I created a new column named 'Memory_Type' and
    # extracted the type of memory using regex. This provided me with a column containing valuable information regarding
    # the type of memory in each device, whether it's SSD, HDD, or Flash Storage.
    df['Memory_Type'] = df['Memory'].str.extract(r'(SSD|HDD|Flash Storage)', expand=False)

    # The 'Memory' column contains values specified in GB and TB. To standardize these values, I converted all of them
    # to GB and removed the units. This process resulted in numeric values all in a single unit of measurement, GB,
    # making the data more consistent and easier to analyze.
    df['Memory_size_GB'] = df['Memory'].apply(convert_to_gb)

    # The 'Cpu' column contains information about both the processor model and its clock frequency. To better organize
    # this data, I split it into two separate columns. Using a lambda function, I separated the processor model and
    # clock frequency at the last space character, resulting in two new columns: 'Processor' for the processor model
    # and 'Clock' for the clock frequency.
    df[['Processor', 'Clock']] = df['Cpu'].apply(lambda x: pd.Series(str(x).rsplit(' ', 1)))


    # To streamline the processor information, I implemented a function called trim_processor to remove less relevant
    # parts of the description. This function checks if the processor description starts with 'Intel'; if so, it retains
    # only the first three words of the description. For other processors, it keeps the first two words.
    df['Processor'] = df['Processor'].apply(trim_processor)

    # To provide better learning models I ograniczyłem number of Processors and changed rare processors to Others
    cpu_counts = df['Processor'].value_counts()
    rare_cpu = cpu_counts[cpu_counts < 10].index
    df['Processor'] = df['Processor'].apply(lambda x: "Others" if x in rare_cpu else x)

    # In the columns 'Clock', 'Ram', and 'Weight', the units are consistent, so I decided to remove the units to obtain
    # numeric values. This was achieved by applying transformation functions to each column.
    df['Clock'] = df['Clock'].apply(lambda x: float(str(x).strip("GHz")))
    df['Ram'] = df["Ram"].apply(lambda x: int(str(x).strip("GB")))
    df['Weight'] = df["Weight"].apply(lambda x: float(str(x).strip("kg")))

    # To disregard non-essential information, I opted to eliminate all detailed GPU information and retain only the brand.
    # This was accomplished by creating a new column 'Gpu_Brand' in the dataframe df. I extracted the brand from the 'Gpu'
    # column by splitting each entry at the first space and keeping only the first part, which typically represents the
    # brand name. This approach simplifies the GPU data to focus solely on the brand, which might be sufficient for modeling purposes.
    df['Gpu_Brand'] = df['Gpu'].str.split(n=1, expand=True)[0]

    # To obtain a numeric value and simplify the screen resolution information, I decided to limit the 'Screen Resolution'
    # data to just the 'Screen Width'. This was done by creating a new column named 'Screen_Width' in the dataframe `df'.
    # I extracted the width from the 'ScreenResolution' column by splitting each entry from the right at the last space,
    # taking the last part which typically contains the resolution (e.g., '1920x1080'), and then splitting this by 'x'
    # to separate width from height. I kept only the width (the first part) and converted it to an integer. This method
    # effectively reduces the screen resolution data to a single, numeric width value, making it more straightforward for
    # numerical analysis or modeling.
    df['Screen_Width'] = df['ScreenResolution'].str.rsplit(' ', n=1).str[-1].str.split('x').str[0].astype(int)

    # After extracting the relevant data, I remove unnecessary columns from the dataframe to streamline it.
    return df.drop(['laptop_ID', 'Product', 'ScreenResolution', 'Cpu', 'Memory', 'Gpu'], axis=1)


def remove_outliers(df):
    """
       Remove outliers from a pandas DataFrame based on the Interquartile Range (IQR) method.

       This function iterates through each numerical column in the DataFrame and removes rows
       that contain outliers. An outlier is defined as a value that is below (Q1 - 1.5 * IQR) or
       above (Q3 + 1.5 * IQR), where Q1 and Q3 are the first and third quartiles, respectively,
       and IQR is the interquartile range (Q3 - Q1).

       Parameters:
       - df (pandas.DataFrame): The DataFrame from which outliers will be removed.

       Returns:
       - pandas.DataFrame: A new DataFrame with outliers removed from the numerical columns.

       Note:
       - This function only considers columns with numerical data.
       - The returned DataFrame may have fewer rows than the input DataFrame due to the removal of outliers.
       - This method is based on the assumption that the data in the numerical columns follows a roughly symmetric distribution. For highly skewed distributions, consider using a different method for outlier detection.
       """
    indices_to_remove = set()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for column in num_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        indices_to_remove.update(outliers_indices)

    return df.drop(indices_to_remove)


def encode_and_scale_features(X_train, X_test, columns_to_encode):
    """
    Encode categorical features and scale numerical features in training and testing datasets.

    This function takes the training and testing datasets along with the names of columns to be
    one-hot encoded. It performs one-hot encoding on the specified columns and standard scaling
    on the numerical columns. The function then combines the encoded and scaled features to
    produce final datasets ready for machine learning model training and evaluation.

    Parameters:
    - X_train (pandas.DataFrame): The training dataset with both numerical and categorical features.
    - X_test (pandas.DataFrame): The testing dataset with both numerical and categorical features.
    - columns_to_encode (list of str): The names of the columns in the datasets to be one-hot encoded.

    Returns:
    - tuple: A tuple containing two pandas.DataFrames:
        - X_train_final: The transformed training dataset with encoded categorical features and scaled numerical features.
        - X_test_final: The transformed testing dataset with encoded categorical features and scaled numerical features.

    Notes:
    - The function uses OneHotEncoder for encoding categorical variables and StandardScaler for scaling numerical variables.
    - It is important that `columns_to_encode` only includes names of categorical columns present in both `X_train` and `X_test`.
    - The function ensures that the transformed datasets retain their original row indices.
    - The scaling is fitted on the training data and then applied to both the training and testing data to prevent data leakage.
    """
    encoder = OneHotEncoder()
    encoder.fit(X_train[columns_to_encode])

    X_train_encoded = encoder.transform(X_train[columns_to_encode])
    X_test_encoded = encoder.transform(X_test[columns_to_encode])

    X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), index=X_train.index)
    X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), index=X_test.index)

    X_train_dropped = X_train.drop(columns=columns_to_encode)
    X_test_dropped = X_test.drop(columns=columns_to_encode)

    scaler = StandardScaler()
    scaler.fit(X_train_dropped)

    X_train_scaled = scaler.transform(X_train_dropped)
    X_test_scaled = scaler.transform(X_test_dropped)

    X_train_final = pd.concat([pd.DataFrame(X_train_scaled, index=X_train.index), X_train_encoded_df], axis=1)
    X_test_final = pd.concat([pd.DataFrame(X_test_scaled, index=X_test.index), X_test_encoded_df], axis=1)

    return X_train_final, X_test_final


def save_evaluation_data(X_test, y_test, X_train=None, y_train=None, prefix=''):
    """
    Save test and optionally training datasets to CSV files with an optional prefix.

    This function saves the testing and, if provided, training feature matrices and target vectors
    to CSV files. Filenames are prefixed with an optional string to distinguish different sets
    or runs.

    Parameters:
    - X_test (pandas.DataFrame): Feature matrix for testing data.
    - y_test (pandas.Series or pandas.DataFrame): Target vector or matrix for testing data.
    - X_train (pandas.DataFrame, optional): Feature matrix for training data. If not provided, training data is not saved.
    - y_train (pandas.Series or pandas.DataFrame, optional): Target vector or matrix for training data. If not provided,
    training data is not saved.
    - prefix (str, optional): A string prefix to prepend to the filenames for differentiation (default is '').

    The function does not return any value. It writes the following files to disk:
    - `<prefix>X_test.csv`: Testing feature matrix.
    - `<prefix>y_test.csv`: Testing target vector/matrix.
    - `<prefix>X_train.csv`: Training feature matrix (if `X_train` and `y_train` are provided).
    - `<prefix>y_train.csv`: Training target vector/matrix (if `X_train` and `y_train` are provided).
    """
    X_test.to_csv(f'eval_data/{prefix}X_test.csv', index=False)
    y_test.to_csv(f'eval_data/{prefix}y_test.csv', index=False)
    if X_train is not None and y_train is not None:
        X_train.to_csv(f'eval_data/{prefix}X_train.csv', index=False)
        y_train.to_csv(f'eval_data/{prefix}y_train.csv', index=False)

def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple regression models, save models and predictions, and evaluation datasets.

    Trains Linear Regression, Lasso Regression, Ridge Regression, Decision Tree Regressor,
    and a Polynomial Regression model on the provided training data. It predicts on the testing data,
    saves models to disk, predictions to text files, and both training and testing datasets to CSV files.

    Parameters:
    - X_train (pandas.DataFrame or numpy.ndarray): Feature matrix for training data.
    - y_train (pandas.Series or numpy.ndarray): Target vector for training data.
    - X_test (pandas.DataFrame or numpy.ndarray): Feature matrix for testing data.
    - y_test (pandas.Series or numpy.ndarray): Target vector for testing data.

    This function does not return any value but has side effects:
    - Saves trained models to `<model_name>_model.joblib`.
    - Saves predictions on the test set to `<model_name>_predictions.txt`.
    - Saves testing and training datasets to CSV files prefixed with 'final_'.

    Notes:
    - The function uses predefined hyperparameters for Lasso, Ridge, and Decision Tree models.
      Adjust these as necessary.
    - Polynomial features expand the feature space for the Polynomial Regression, possibly increasing memory usage.
    - Requires `joblib` for model serialization and `numpy.savetxt` for predictions.
    - After training and predictions, it calls `save_evaluation_data` to save datasets, facilitating
      evaluation or further experiments.
    """
    models = {
        'linear': LinearRegression(),
        'lasso': Lasso(alpha=1.2),
        'ridge': Ridge(alpha=1.0),
        'tree': DecisionTreeRegressor(max_depth=3, min_samples_split=6),
    }

    poly_features = PolynomialFeatures(degree=2)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    poly_model = LinearRegression()
    models['poly'] = poly_model

    for name, model in models.items():
        if name == 'poly':
            model.fit(X_train_poly, y_train)
            joblib.dump(model, f'models/{name}_model.joblib')
        else:
            model.fit(X_train, y_train)
            joblib.dump(model, f'models/{name}_model.joblib')

    save_evaluation_data(X_test, y_test, X_train, y_train, prefix='final_')


if __name__ == "__main__":
    source_file = 'laptop_price.csv'
    df = read_and_preprocess_data(source_file)

    X = df.drop('Price_euros', axis=1)
    y = df['Price_euros']

    columns_to_encode = ['Company', 'TypeName', 'OpSys', 'Memory_Type', 'Gpu_Brand', "Processor"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_final, X_test_final = encode_and_scale_features(X_train, X_test, columns_to_encode)

    train_models(X_train_final, y_train, X_test_final, y_test)
