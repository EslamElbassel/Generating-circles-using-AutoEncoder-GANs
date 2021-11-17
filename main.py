import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Input, Model
from keras.layers import Dense
from keras.layers import LeakyReLU

# _____Loading circles data from csv file_____
points = pd.read_csv("Geom(1).csv")

# _____Splitting data into train and test_____
train_X = points.iloc[:17]
test_X = points.iloc[17:]

# _____Creating the models_____

# creating the autoencoder
code_size = 60  # 30 # size of encoded image(compressed from 122 value)
input_points = Input(shape=(122,))
encoded = Dense(code_size, activation=LeakyReLU(alpha=0.1))(input_points)
decoded = Dense(122, activation=LeakyReLU(alpha=0.1))(encoded)
autoencoder = Model(inputs=input_points, outputs=decoded)

# creating the encoder(which map the input image to its code)
encoder = Model(inputs=input_points, outputs=encoded)

# creating the decoder(which map the generated code of image to its reconstruction)
decoder_input = Input(shape=(code_size,))
decoder_output = autoencoder.layers[-1](decoder_input)  # getting the last layer of the autoencoder model
decoder = Model(inputs=decoder_input, outputs=decoder_output)

autoencoder.compile(loss='mean_squared_error', optimizer='adam')

# Training the autoencoder model
autoencoder.fit(train_X, train_X, batch_size=256, epochs=300)

# Decode and Encode unseen circle points(testing the models)
encodeder_output = encoder.predict(test_X)
decoder_output = decoder.predict(encodeder_output)

# _____Visualize the reconstructed circle against their original circle_____

print("Max value", points.max().max())  # maximum  point value
n = 5  # number of circles
plt.figure(figsize=(15, 10))
for i in range(n):
    # Plot original circles
    original_X, original_y = test_X.iloc[i][::2], test_X.iloc[i][1::2]
    plt.subplot(2, n, i + 1)
    plt.xlim([-150, 150])
    plt.ylim([-150, 150])
    plt.title("Original")
    plt.scatter(original_X, original_y, color='teal')

    # Plot reconstructed circles
    reconstructed_X, reconstructed_y = decoder_output[i][::2], decoder_output[i][1::2]
    plt.subplot(2, n, i + 1 + n)
    plt.xlim([-150, 150])
    plt.ylim([-150, 150])
    plt.title("Reconstructed")
    plt.scatter(reconstructed_X, reconstructed_y, color='brown')

plt.show()

# _____Calculate the error(MSE)_____
print(autoencoder.evaluate(decoder_output, test_X))
print("MSE: ", mean_squared_error(test_X, decoder_output))
