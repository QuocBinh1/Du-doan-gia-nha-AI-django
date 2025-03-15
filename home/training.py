import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import os

data = pd.read_csv('USA_Housing.csv')
data = data.drop('Address', axis = 1)

    #tách dữ liệu thành các biến đầu vào và biến đầu ra
x = data.drop('Price', axis = 1).values
y = data['Price'].values

    #chia duữ liệu tành tập huấn luyện và tập kiểm tra(70% dữ liệu để huấn luyện, 30% dữ liệu để kiểm tra)
train_size = int(0.7 * len(x))
x_train , x_test = x[:train_size], x[train_size:]
y_train , y_test = y[:train_size], y[train_size:]

    #thêm cột hệ số chahnwj vào x_train 
x_train  = np.hstack((np.ones((x_train.shape[0], 1)), x_train))


    #tính toansc ác hệ số hồi quy beta = (X^T * X)^-1 * X^T * y
XTX = np.dot(x_train.T, x_train)
XTX_inv = np.linalg.inv(XTX)
XTY = np.dot(x_train.T, y_train)
beta = np.dot(XTX_inv, XTY)

    #hàm dự đoán giá nhà dưa trên các giá trị đầu vào
    
    #giả sử bạn nhận cá giá trị từ request GEt
var1 = float(request.GET['n1'])
var2 = float(request.GET['n2'])
var3 = float(request.GET['n3'])
var4 = float(request.GET['n4'])
var5 = float(request.GET['n5'])
    
    
def predict_print(var1,var2,var3,var4,var5):
    input_vars = np.array([1 , var1 ,var2,var3 ,var4,var5]) #thêm hệ số chặn
    return np.dot(input_vars,beta)

pred = predict_print(var1 , var2,var3,var4,var5)
pred = round(pred)

price = "giá dự đoán là $ " + str(pred)
print(price)