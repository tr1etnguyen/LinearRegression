import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from pyod.models.knn import KNN   # kNN detector
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Lasso
#Tải dữ liệu từ file vào chương trình
df = pd.read_csv("Housing.csv")

# X = df.drop(columns="price")

# #print(x.head())

# y = df["price"]

# X['mainroad'] = X['mainroad'].map({'yes': 1, 'no': 0})
# X['guestroom'] = X['guestroom'].map({'yes': 1, 'no': 0})
# X['basement'] = X['basement'].map({'yes': 1, 'no': 0})
# X['hotwaterheating'] = X['hotwaterheating'].map({'yes': 1, 'no': 0})
# X['airconditioning'] = X['airconditioning'].map({'yes': 1, 'no': 0})
# X['prefarea'] = X['prefarea'].map({'yes': 1, 'no': 0})
# X['furnishingstatus'] = X['furnishingstatus'].map({'unfurnished':0,'semi-furnished':1,'furnished':2})

def scatter_subplot(target_var, df, num_cols, title):
    """
    Vẽ biểu đồ phân tán giữa các biến số và biến mục tiêu dưới dạng subplot.

    Parameters:
        target_var (str): Tên cột mục tiêu.
        df (pd.DataFrame): Dữ liệu.
        num_cols (list): Danh sách các cột số để vẽ scatter plot với target.
        title (str): Tiêu đề của toàn bộ biểu đồ.
    """
    feature = [col for col in df.columns if col != target_var]
    num_plot = len(feature)
    num_rows = int(np.ceil(num_plot/num_cols))
    

    fig, axes = plt.subplots(nrows=num_rows ,ncols= num_cols, figsize=(20, 10*num_rows)) #Trả về (fig) và axes 2 chiều 
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()  # làm phẳng để dễ duyệt mảng

    for i, nf in enumerate(feature):
        axes[i].scatter(df[nf], df[target_var], alpha=0.8, color='blue')
        axes[i].set_xlabel(nf)
        axes[i].set_ylabel(target_var)
        axes[i].set_title(f'{nf} vs {target_var}')
        #axes[i].grid(True)

    # Ẩn subplot dư (nếu có)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # chừa chỗ cho suptitle
    plt.show()
    
def boxplot(df, num_cols, title, fig_no):
  '''
  Vẽ biểu đồ hộp để xác định outliers
  '''
  num_plots = len(df.columns)
  num_rows = int(np.ceil(num_plots / num_cols))  # Calculate required rows

  fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))
  fig.suptitle(title, fontsize=16)
  axes = axes.flatten()  # Flatten axes array for easy iteration

  # Tạo boxplot cho mỗi cột
  for i, col in enumerate(df.columns):
    #sns.boxplot(y=df[col], ax=axes[i], color='skyblue')
    axes[i].boxplot(df[col], vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    axes[i].set_title(f"Fig.{fig_no}.{i+1} - {col}", fontsize=12)
    axes[i].set_ylabel('')  # Remove y-axis label for cleaner look

  # Ẩn subplot dư (nếu có)
  for j in range(i + 1, len(axes)):
    axes[j].axis('off')

  plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
  plt.show()
# Check dataset info
#print(df.info())

#Chuyển đổi dữ liệu các cột int64 sang float
df[df.select_dtypes(include=['int64']).columns] = df.select_dtypes(include=['int64']).astype(float)

#Xử lí dữ liệu yes/no
binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))

#Xử lí dữ liệu thuộc đặc trưng furnishing 
furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
df['furnishingstatus'] = df['furnishingstatus'].map(furnishing_map)

#print(df.info())
# Kiểm tra giá trị null
#print(df.head())

# Feature Engineering
df.loc[:, 'price_per_area'] = df.loc[:, 'price'] / df.loc[:, 'area'] #Sự phụ thuộc giữa giá nhà và diện tích
df.loc[:,'price_per_bedroom'] = df.loc[:,'price'] / df.loc[:,'bedrooms'] #........giá nhà và số lượng phòng ngủ @@
df.loc[:,'price_per_bathroom'] = df.loc[:,'price'] / df.loc[:,'bathrooms'] #...........Phòng tắm
df.loc[:,'price_per_story'] = df.loc[:,'price'] / df.loc[:,'stories'] #...................Tầng (US English @@)
luxury_score = df['airconditioning'] + df['hotwaterheating'] + df['prefarea'] + df['furnishingstatus'] + df['mainroad']
df.loc[:,'price_per_score'] = df.loc[:,'price'] / luxury_score

#Tổng số phòng
total_room = df.loc[:,'bedrooms'] + df.loc[:,'bathrooms'] + df.loc[:,'guestroom'] + df.loc[:,'basement']
df.loc[:,'price_per_room'] = df.loc[:,'price'] / total_room

#Biểu đồ Scatter
# title = 'Fig.1 - Scatter Plots: Price_per_Area vs Independent Variables'
# target_var = 'price_per_area'
# num_cols = 6
#Vẽ biểu đồ phân tán để dễ xác định outliers
#scatter_subplot(target_var, df, num_cols, title)

#Chuyển đổi dữ liệu
#Cột muốn chuyển đổi
columns= ['price','area','price_per_area','price_per_bedroom','price_per_bathroom','price_per_story','price_per_score','price_per_room']

# Tạo một bản sao khác
df_transformed = df.copy()

# Square Root để tránh tạo ra sự chênh lệch giữa các đặc trưng quá lớn
df_transformed[columns] = np.log(df[columns])

# ghi đè các giá trị inf với nan
df_transformed = df_transformed.replace([np.inf, -np.inf], np.nan)
df_cleaned = df_transformed.dropna() # Xóa các điểm dữ liệu có giá trị nan
#print(df_cleaned.head())
num_cols = 4
title = 'Fig.3 - Boxplots of Numerical Features'
fig_no = 3
#boxplot(df_cleaned, num_cols, title, fig_no)

data_outliers_detection = df_cleaned.copy()
clf_name = 'KNN'

np.random.seed(42)
#Dùng K-Nearest Neighbors để tìm biến ngoại lai
clf = KNN()
clf.fit(data_outliers_detection)

outliers, outliers_confidence = clf.predict(data_outliers_detection, return_confidence=True)  # outlier labels (0 or 1) và khoảng tin cậy ở [0,1]
#print(outliers_confidence)
# Tạo Df để đánh dấu dữ liệu ngoại lai
results_outliers = data_outliers_detection.copy() 
results_outliers['Outlier'] = outliers  
results_outliers['Confidence'] = outliers_confidence  

#print(results_outliers.head())

data_outliers_detection['Outlier'] = outliers


# Loại bỏ ngoại lai
no_outliers_df = data_outliers_detection[data_outliers_detection['Outlier'] == 0]
no_outliers_df = no_outliers_df.drop(columns="Outlier")
# print("Data Without Outliers:")
# print(no_outliers_df.info())
#Check lại tập dữ liệu đã loại bỏ outliers
num_cols = 4
title = 'Fig.4 - Recheck Boxplots of Features'
fig_no = 4
#boxplot(no_outliers_df, num_cols, title, fig_no)

#PCA 
pca_data = no_outliers_df.drop(columns=['price'])

# Khởi tạo PCA
prepro = StandardScaler()
#pca_data_scaled = preprocessing.scale(pca_data) #scale cho việc sử dụng PCA
pca_data_scaled = prepro.fit_transform(pca_data)
pca = PCA(len(pca_data.columns))
pca.fit(pca_data_scaled)

# Kết quả
pr_var = pca.explained_variance_ratio_
cum_pr = np.cumsum(pca.explained_variance_ratio_)
ind = ['Proportion of variance','Cumulative Proportion of variance']

# Số lượng PC 
num_components = len(pca_data.columns)


col = [f'PC{i}' for i in range(num_components)]

#print(pd.DataFrame(np.vstack((pr_var, cum_pr)), ind, columns = col))

# Hệ số (Coefficients) cho PC
pc_res = pd.DataFrame(pca.components_.T, index=list(pca_data.columns),columns=col)
#print(pc_res)
# Tìm hệ số tuyệt đối cao nhất cho mỗi đặc trưng
highest_coefficients = pc_res.abs().idxmax(axis=0)
highest_values = pc_res.max(axis=0)
# Xuất kết quả PC
result = pd.DataFrame({
    'Highest_PC': highest_coefficients,
    'Value': highest_values
})
#print(result)
########################
# Dựa vào kết quả PC lấy 10 đặc trưng cao nhất
top_10_unique_highest_pc = result['Highest_PC'].drop_duplicates().head(10)
#print(top_10_unique_highest_pc.values)

###
# định nghĩa nhãn y và tập các dữ liệu đặc trưng X
X = no_outliers_df[top_10_unique_highest_pc]
y = no_outliers_df['price']
#print(y.tail)
#Chuẩn hóa các đặc trưng features
scaler = StandardScaler()
#print(X.index)
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
#print(X_scaled.tail())

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Mô hình linear regression 
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)
# Đảo ngược log để lấy lại giá nhà thực
y_pred_real = np.exp(y_pred)
y_test_real = np.exp(y_test)
# Đánh giá
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred_real)
print(f"MAE  = {mae:.2f}")
print(f"MSE  = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.4f}")
##################Test
# # Điểm dữ liệu đầu vào
# x_test = [1750000	,3850	,3	,1	,2	,'yes',	'no',	'no',	'no',	'no',	'0',	'no',	'unfurnished']

# # Tên cột theo đúng thứ tự trong dữ liệu gốc
# columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories',
#            'mainroad', 'guestroom', 'basement', 'hotwaterheating',
#            'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

# # Tạo DataFrame từ x_test
# df_test = pd.DataFrame([x_test], columns=columns)

# # Tiền xử lý giống như dữ liệu huấn luyện
# binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
# df_test[binary_columns] = df_test[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))
# df_test['furnishingstatus'] = df_test['furnishingstatus'].map({'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2})

# # Tính các đặc trưng mới
# df_test['price_per_area'] = df_test['price'] / df_test['area']
# df_test['price_per_bedroom'] = df_test['price'] / df_test['bedrooms']
# df_test['price_per_bathroom'] = df_test['price'] / df_test['bathrooms']
# df_test['price_per_story'] = df_test['price'] / df_test['stories']
# luxury_score = df_test['airconditioning'] + df_test['hotwaterheating'] + df_test['prefarea'] + df_test['furnishingstatus'] + df_test['mainroad']
# df_test['price_per_score'] = df_test['price'] / luxury_score
# total_room = df_test['bedrooms'] + df_test['bathrooms'] + df_test['guestroom'] + df_test['basement']
# df_test['price_per_room'] = df_test['price'] / total_room

# # Chuyển sang log giống như dữ liệu huấn luyện
# log_cols = ['price','area','price_per_area','price_per_bedroom','price_per_bathroom','price_per_story','price_per_score','price_per_room']
# df_test[log_cols] = np.log(df_test[log_cols])

# # Chọn đúng các đặc trưng đã chọn từ PCA
# X_test_input = df_test[top_10_unique_highest_pc]

# # Chuẩn hóa bằng scaler đã fit trước đó
# X_test_input_scaled = scaler.transform(X_test_input)
# X_test_input_scaled_df = pd.DataFrame(X_test_input_scaled, columns=X_test_input.columns)
# print(X_test_input_scaled_df)


# # Dự đoán
# y_test_pred = model.predict(X_test_input_scaled_df)
# print(y_test_pred)
# print(f"Giá nhà dự đoán: {np.exp(y_test_pred[0]):,.0f}")




# Đánh giá
# MAE  = 592006.15
# Trung bình mỗi dự đoán lệch khoảng 592006.115 so với giá trị thực tế
# MSE  = 683399283147.60
# RMSE = 826679.67
# Sai số bình phương trung bình gốc, cao hơn MAE, cho thấy vẫn còn một số điểm sai số lớn ảnh hưởng đáng kể
# R²   = 0.8141
# Có thể hiểu là mô hình giải thích được khoảng 81.4% phương sai của dữ liệu thực tế, đây là một kết quả khá tốt


