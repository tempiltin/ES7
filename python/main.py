import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Ma'lumotlarni yaratish
data = pd.read_csv("student_risk_data.csv")

# X va y ni aniqlash
X = data[['Baho', 'Davomat', 'SocioEconomic1', 'SocioEconomic2', 'Boshqa']]
y = data['Risk_of_Failing']

# Ma'lumotlarni o'qitish va test to'plamlariga ajratish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1) Chiziqli regressiya modeli asosida
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# 2) Ko'p o'lchovli chiziqli regressiya modeli asosida
multi_linear_model = LinearRegression()
multi_linear_model.fit(X_train, y_train)
multi_linear_predictions = multi_linear_model.predict(X_test)

# 3) Polinomil regressiya modeli asosida
degree = 3  # Polinom darajasi
poly = PolynomialFeatures(degree=degree)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)
poly_predictions = poly_model.predict(X_poly_test)

# Grafik tasvirlash
plt.figure(figsize=(18, 5))

# Chiziqli regressiya grafikasi
plt.subplot(1, 3, 1)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Asl qiymatlar', alpha=0.6)
plt.plot(range(len(linear_predictions)), linear_predictions, color='red', label='Bashorat', alpha=0.8)
plt.title("Chiziqli Regressiya")
plt.xlabel("Namuna")
plt.ylabel("Risk of Failing")
plt.legend()

# Ko'p o'lchovli regressiya grafikasi
plt.subplot(1, 3, 2)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Asl qiymatlar', alpha=0.6)
plt.plot(range(len(multi_linear_predictions)), multi_linear_predictions, color='green', label='Bashorat', alpha=0.8)
plt.title("Ko'p O'lchovli Regressiya")
plt.xlabel("Namuna")
plt.ylabel("Risk of Failing")
plt.legend()

# Polinomil regressiya grafikasi
plt.subplot(1, 3, 3)
plt.scatter(range(len(y_test)), y_test, color='blue', label='Asl qiymatlar', alpha=0.6)
plt.plot(range(len(poly_predictions)), poly_predictions, color='purple', label='Bashorat', alpha=0.8)
plt.title("Polinomil Regressiya")
plt.xlabel("Namuna")
plt.ylabel("Risk of Failing")
plt.legend()

# Grafiklarni ko'rsatish
plt.tight_layout()
plt.show()
