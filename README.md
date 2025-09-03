House Price Prediction using Linear Regression

This project demonstrates how to build a simple linear regression model to predict house prices based on house sizes using Python and scikit-learn. The project includes data visualization, model training, and performance evaluation using the R² score.
Dataset

The code expects a file named home_dataset.csv in the same directory. The dataset should have the following columns:
HouseSize — Size of the house in square feet (numeric)
HousePrice — Price of the house in British pounds (£) (numeric)

Example structure:
HouseSize	  HousePrice
1400	      300000
1600	      350000


Features:
- Data visualization using Matplotlib and Seaborn
- Splitting dataset into training and testing sets
- Linear regression model training and prediction using scikit-learn
- Visual comparison of actual vs. predicted house prices
- Model performance evaluation using R² score
  
Output Plots:
1. Scatter Plot — Visualizes the relationship between house size and price.
2. Prediction Plot — Compares actual prices with predicted prices using a regression line.
3. R² Score — Displays how well the model explains the variance in the data.

Libraries Used:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
  
Install them with:
pip install numpy pandas matplotlib seaborn scikit-learn

How to Run
Ensure home_dataset.csv is in the same folder.

Run the script:
python your_script_name.py

Sample R² Score Output:
The R² score is displayed on the prediction plot. It shows the proportion of variance in the house prices that is predictable from house size.
R² = 0.873
A higher R² (closer to 1) means a better fit.

Example Visualization
<img width="997" height="657" alt="Screenshot 2025-09-03 at 11 30 33" src="https://github.com/user-attachments/assets/dd9fcee6-127c-4088-bdbc-7528f893ddf1" />
<img width="1002" height="661" alt="Screenshot 2025-09-03 at 11 31 36" src="https://github.com/user-attachments/assets/b2b28891-6a62-4f32-816f-b0a57016c280" />

Contact:
For questions or suggestions, feel free to open an issue or pull request.
