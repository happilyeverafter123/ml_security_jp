import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
file_path = os.environ.get("FILE_PATH")
with open(file_path) as f:
    content = f.readlines()
    
content = [x.strip() for x in content]

data = pd.DataFrame()

data['feature'] = content

# print(data['feature'].head(10))
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np

max_words = 800
max_len = 100

X = data['feature']

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=max_len, truncating='post')

print(tokenizer.word_index)

print(X[0])

labels_path = os.environ.get("LABELS_PATH")
with open(labels_path) as f:
    label_data = f.readlines()
    
label_data = [x.strip() for x in label_data]

data['labels'] = label_data

#import matplotlib.pyplot as plt

#plt.figure(figsize=(20, 10))
#sns.countplot(data['labels'])

#output_path = "/home/u634567g/for_security/ch04/label_distribution.png"
#plt.savefig(output_path, format='png', dpi=300)

y = data['labels'].apply(lambda x: 1 if x == 'Virus' else 0)

print(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# def malware_model():
#     model = tf.keras.Sequential()    
#     model.add(tf.keras.layers.Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
#     model.add(tf.keras.layers.LSTM(32, return_sequences=False))
#     model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#     return model

# model = malware_model()
# print(model.summary())
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        clear_session = tf.keras.backend.clear_session
        clear_session()
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
        model.add(tf.keras.layers.LSTM(32, return_sequences=False))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            beta_1=trial.suggest_uniform("beta_1", 0.0, 1.0),
            beta_2=trial.suggest_uniform("beta_2", 0.0, 1.0)
        )
        
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Train the model
        model.fit(self.X, self.y, epochs=10, batch_size=256, validation_data=(X_test, y_test), verbose=0)
        
        return model.evaluate(X_test, y_test, verbose=0)[1]
    
import optuna

objective = Objective(X_train, y_train)
study = optuna.create_study()
study.optimize(objective, timeout=1200)

from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasClassifier

def buildmodel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=max_words, output_dim=300, input_length=max_len))
    model.add(tf.keras.layers.LSTM(32, return_sequences=False))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=study.best_params['learning_rate'],
        beta_1=study.best_params['beta_1'],
        beta_2=study.best_params['beta_2']
    )
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

clf = KerasClassifier(model=buildmodel, epochs=10, batch_size=64, verbose=1)

results = cross_val_score(clf, X_train, y_train, cv=5)

print("Cross-validation results: ", results.mean())
print("Best parameters: ", study.best_params)
print("Best trial: ", study.best_trial)
