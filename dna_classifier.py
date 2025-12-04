"""
2. Load the data and preprocess it appropriately so that it can be input into your model.

3. Train a neural network on the dataset, splitting the data into a training and test set at a 20/80 split (testing/training respectively). 

4.  Reach at least s 90% accuracy rate using the tensorflow accuracy metric. 
"""

import pandas as pd # import necessary libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import plotly.express as px


data=pd.read_csv("classification_and_seqs_aln.csv") # read the data file
print(data.head())

# A and T related closely. C and G related closely. Encode so that the numbers are close together

seq=data["sequence"] # Set initial values for variables and arrays
encoder=LabelEncoder()
species_encoder=LabelEncoder()
encoded=""
temp=[]
X=[]
print(seq[0])
print("Hello", seq[0][0])

#encoder=encoder.fit(list(seq[0])) # Fit the encoder to recognize the characters given in each sequence
encoder=encoder.fit(["-", "A", "T", "C", "G"]) # fits the encoder (makes a "key" for the characters in the DNA sequence)

for i in range(480): # Repeat for each DNA sequence 
    temp=[]
    encoded=encoder.transform(list(seq[i])) # Encode the current sequence
    for n in range(len(seq[i])):
        temp.append(encoded[n]) # Make an array of each encoded character for the current sequence
    X.append(temp) # Add the current array to the array X to make a 2D array
# encode each character, then take each numerical value as a feature
# X should be 2d array


X=np.array(X) # can ignore read id
print(data["species"])
y=species_encoder.fit_transform(data["species"]) # Set the species encoder and label encode the species of bacteria
print(y)
print(X[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42) # Split the data

model = tf.keras.models.Sequential([ # Define the model
    tf.keras.layers.Dense(len(seq[0]), input_shape=[len(seq[0])]),
    tf.keras.layers.Dense(44, activation="relu"),
    #tf.keras.layers.Dense(18, activation="relu"),
    #tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(39, activation="relu"), # 34 nodes for 34 different species (34 different output nodes)
    tf.keras.layers.Softmax() # Set later for species of DNA
])

lr=0.0001 # Define the learning rate

model.compile( # Compile the model
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=["accuracy"]
)

history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100) # Fit and run the model

df=pd.DataFrame(history.history)["loss"] # Plot the loss graph. "Index" = "# of Epochs"
plot1=px.scatter(df)
plot1.show(renderer="browser")


# Last val_accuracy value was 0.9062 > .9 

"""
What different settings did you experiment with, and how did each one affect your model’s performance?
Some of the settings that I experimented with include changing the number of hidden layers, the number of nodes in each
hidden layer, the activation function of each layer, and the learning rate. I found that, to some extent, as the number of 
hidden layers, the number of nodes, and the learning rate increases, val_accuracy decreases due to either overfitting or 
too drastic changes in weights and biases during backpropagation. The opposite effect occurs when the number of hidden 
layers, the number of nodes, or the learning rate decreases. The activation function has varying affects on val_accuracy
depending on the given function. 

Describe which choices ultimately worked best, which did not, and provide reasoning for why you think those outcomes occurred.
Using a considerably low learning rate worked best for my model because it allowed the model to gradually work towards
values for weights and biases that best minimized loss/maximized val_accuracy. Using only relu activation functions worked
best, which could be due to the fact that other activation functions did not fit the data well and could have been 
influenced by issues such as a vanishing gradient. Using a moderately high number of nodes worked best because it made
the model complex enough to handle the many input features but general enough to prevent overfitting. Using a low number
hidden layers worked best because it limited overfitting, which was a problem that I ran into throughout the training
process. 
"""