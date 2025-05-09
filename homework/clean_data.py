import nltk
import pandas as pd


def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

def create_key(df):
    df = df.copy()
    df["key"] = df["raw_text"]
    df["key"] = df["key"].str.strip()
    df["key"] = df["key"].str.lower()
    df["key"] = df["key"].str.translate(str.maketrans("", "", "!\"#$%&'()*+,-./:;<=>?@[\\]_^{|}~/"))
    df["key"] = df["key"].str.split()
    
    stemmer = nltk.PorterStemmer()

    df["key"] = df["key"].apply(lambda x: [stemmer.stem(word) for word in x])
    df["key"] = df["key"].apply(lambda x: sorted(set(x)))
    df["key"] = df["key"].str.join(" ")

    return df


def generate_cleaned_column(df):
    df = df.copy()
    keys = df.drop_duplicates(subset="key",keep="first")    
    key_dict = dict(zip(keys["key"], keys["raw_text"]))
    df["cleaned"] = df["key"].map(key_dict)

    return df


def save_data(df, output_file):
    df = df.copy()
    df = df[["cleaned"]]
    df = df.rename(columns={"cleaned": "cleaned_text"})
    df.to_csv(output_file, index = False)

def main(input_file, output_file):
    df = load_data(input_file)
    df = create_key(df)
    df = generate_cleaned_column(df)
    print(df)
    df.to_csv("./files/test.csv", index=False)
    save_data(df, output_file)

if __name__ == "__main__":
    main("./files/input.txt", "./files/output.txt")