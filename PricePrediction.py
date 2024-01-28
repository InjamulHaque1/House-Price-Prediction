import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file_path = "C:/Users/Navid/Desktop/MachineLearning/Datasets/melb_data.csv"
house_data = pd.read_csv(file_path)

house_data = house_data.dropna(axis=0)

y = house_data['Price']
house_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = house_data[house_features]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

house_model = DecisionTreeRegressor(random_state=1)
house_model.fit(x_train, y_train)

x_first_200 = x.head(300)
predictions_first_200 = house_model.predict(x_first_200)

r_squared = r2_score(y.head(300), predictions_first_200)

plt.plot(y.head(300).values, label='Actual Prices', linestyle='-', marker='o', markersize=5)
plt.plot(predictions_first_200, label=f'Predicted Prices (Accuracy: {r_squared:0.4f})', linestyle='-', marker='o', markersize=5)
plt.xlabel('Houses (First 100)')
plt.ylabel('Prices')
plt.title('Actual Prices vs. Predicted Prices for the First 300 Houses')
plt.legend()
plt.show()
