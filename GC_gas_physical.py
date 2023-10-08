import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

# 构建简单的ResNet模型类
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)  # 线性输出层

    def forward(self, x):
        return self.resnet(x)
    
    # 数据二维化
    def To2D(self, data):
        mid = []
        X = np.array(data.iloc[:, 4:])
        for x in X:
            mid.append(np.outer(x, x))
        return np.array(mid)
            
    # 绘制热力图
    def show(self, data, id = 0):
        if "To2d" in data.columns:
            print("The To2d is exist, is generating figures...")
        else:
            raise ValueError("The To2d does not exist in DataFrame.")
            
        plt.rcParams['font.family'] = 'SimHei'
        plt.imshow(data.To2d[id], cmap='Reds', interpolation='nearest')
        plt.colorbar() 
        plt.title("{}".format(data.Type[id]))
        plt.show() 
    
    # 数据处理
    def data_deal(self, GC, RON, num = 4, batch = 32):
        id_test = np.arange(1, len(GC), num)
        id_train = np.setdiff1d(torch.arange(0, len(GC)), id_test)
        GC_test = torch.tensor(GC[id_test], dtype = torch.float32).reshape(-1, 1, 305, 305)
        GC_train =  torch.tensor(GC[id_train], dtype = torch.float32).reshape(-1, 1, 305, 305)
        RON_test = torch.tensor(RON[id_test], dtype = torch.float32).reshape(-1, 1)
        RON_train = torch.tensor(RON[id_train], dtype = torch.float32).reshape(-1, 1)
        set_test = TensorDataset(GC_test, RON_test)
        set_train = TensorDataset(GC_train, RON_train)
        Test = DataLoader(set_test, batch_size = batch, shuffle = True)
        Train = DataLoader(set_train, batch_size = batch, shuffle = True)
        return Train, Test
    
    # 训练模型
    def train_resnet(self, model, dataloader, num_epochs = 20):
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))  # 调整targets的维度
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader)}')
        return model
    
    # 在测试集上评估模型
    def predict_resnet(self, model, dataloader):
        model.eval()
        total_loss = 0
        all_inputs = []
        all_outputs = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = model(inputs)
                all_inputs.append(targets)
                all_outputs.append(outputs)
                loss = criterion(outputs, targets.view(-1, 1))
                total_loss += loss.item()
        average_loss = total_loss / len(dataloader)
        print(f'Average Test Loss: {average_loss:.4f}')        
        
        
class GC_Model(object):
    
    def __init__(self, physical_name = "RON", moleculer_column = 5, 
                 alpha_1 = 1e-6, alpha_2 = 1e-6):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.physical_name = physical_name
        self.moleculer_column = moleculer_column
    
    # 数据导入
    def data_read(self, filename):
        # excel导入
        data = pd.DataFrame(pd.read_excel(filename))
        data_cg = data.sort_values(by = [self.physical_name, "ID"], ascending = True)
        return data, data_cg
            
    def train_test_extend(self, data, num = 10):
        data = data.sort_values(by = [self.physical_name, "ID"], ascending = True)
        RON = np.array(data[self.physical_name])
        Type = np.array(data.Name)
        GC = np.array(data.iloc[:, self.moleculer_column:])
        train_x, test_x, train_y, test_y, train_type, test_type = (GC[num:-num], 
                                                            np.vstack((GC[:num], GC[-num:])), 
                                                            RON[num:-num], 
                                                            np.hstack((RON[:num], RON[-num:])),
                                                            Type[num:-num], 
                                                            np.hstack((Type[:num], Type[-num:]))
                                                            )
        return train_x, test_x, train_y, test_y, train_type, test_type

    def train_test_intend(self, data, num = 5):
        data = data.sort_values(by = [self.physical_name, "ID"], ascending = True)
        RON = np.array(data[self.physical_name])
        Type = np.array(data.Name)
        GC = np.array(data.iloc[:, self.moleculer_column:])
        id_test = np.arange(1, GC.shape[0]-1, num)
        id_train = np.setdiff1d(np.arange(0, GC.shape[0]), id_test)
        train_x, test_x, train_y, test_y, train_type, test_type = (GC[id_train], 
                                                            GC[id_test], 
                                                            RON[id_train], 
                                                            RON[id_test],
                                                            Type[id_train], 
                                                            Type[id_test]
                                                            )
        return train_x, test_x, train_y, test_y, train_type, test_type

    def try_different_method(self, data, train_test_split = "intend", num = 5):
        # 数据分割
        if train_test_split == "extend":
            train_x, test_x, train_y, test_y, train_type, test_type = self.train_test_extend(data, num)
        else:
            train_x, test_x, train_y, test_y, train_type, test_type = self.train_test_intend(data, num)
        
        model = linear_model.BayesianRidge(alpha_1=1e-6, alpha_2=1e-6)
        model.fit(train_x, train_y)
        result = model.predict(test_x)
        plt.figure(figsize=(15, 5))
        y = test_y;
        plt.plot(np.arange(len(result)), y, 'go-', label='true value')
        plt.plot(np.arange(len(result)), result, 'ro-', label='predict value')
        plt.title('mse: %f' %np.mean(np.abs(y-result)))
        plt.legend()
        plt.show() 
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(result)), np.abs(y-result), 'go-', label='true value')
        plt.title('mse: %f' %np.mean(np.abs(y-result)))
        plt.legend()
        plt.show()
        return (pd.DataFrame(np.vstack((test_type, y, result, abs(y-result),))).T,
                pd.DataFrame(np.vstack((train_type, train_y, model.predict(train_x), 
                           abs(train_y-model.predict(train_x))))).T,
                np.vstack((np.mean(np.abs(y-result)), max(np.abs(y-result)), 
                                sum(np.abs(y-result)>0.5), sum(np.abs(y-result)>1))),
                np.vstack((np.mean(np.abs(train_y-model.predict(train_x))), max(np.abs(train_y-model.predict(train_x))), 
                                sum(np.abs(train_y-model.predict(train_x))>0.5), sum(np.abs(train_y-model.predict(train_x))>1))),
                model.coef_, model.intercept_)
    
    def predict(self, data, coef_, intercept_):
        X = np.array(data.iloc[:, self.moleculer_column:])
        res = np.dot(X, coef_) + intercept_
        return res
    
    def cross_val(self, data):
        data = data.reset_index(drop=True)
        intercept_ = []
        coef_ = []
        for i in range(len(data)):
            id_test = i
            id_train = np.setdiff1d(np.arange(len(data)), i)
            test_x, train_x, test_y, train_y, test_type, train_type = (data.iloc[id_test, self.moleculer_column:],
                                                                       data.iloc[id_train, self.moleculer_column:],
                                                                       data.RON[id_test],
                                                                       data.RON[id_train],
                                                                       data.Type[id_test],
                                                                       data.Type[id_train]
                                                                       )
            test_x = np.array(test_x).reshape(1, -1)
            sol = self.try_different_method(train_x, train_y, test_x, test_y, train_type, test_type, self.moleculer_column)
            intercept_.append(sol[4])
            coef_.append(sol[3])
        return np.array(intercept_), np.array(coef_)
            
            

# =============================================================================
# CNN code
# =============================================================================

torch.manual_seed(123)

# 数据导入
filename = r"E:\Syspetro\_3_汽油快评\_16_快评工作交接\8.算法对接\3.物性计算算法\训练参数_终馏点\FBP_石脑油.xlsx"
RON = GC_Model()
data, data_cg = RON.data_read(filename, 'FBP')

# 定义损失函数和优化器
model = ResNetModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #weight_decay=0.001

# 二维化
data_2d = model.To2D(data.iloc[283:389])
ron = np.array(data.RON[283:389])
Train, Test = model.data_deal(data_2d, ron)


# 定义损失函数和优化器
model = ResNetModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) #weight_decay=0.001

summary(model, input_size = (1, 305, 305))

# 模型训练        
model.train(model, Train, 100)

# 模型测试
model.predict_resnet(model, Test)




# =============================================================================
# 
# =============================================================================
filename = r"E:\Syspetro\_3_汽油快评\_16_快评工作交接\8.算法对接\3.物性计算算法\训练参数_辛烷值alpha模型\T2_烷基化.xlsx"
RON = GC_Model('RON', 5)
data, data_cg = RON.data_read(filename)
sol_1 = RON.try_different_method(data.iloc[177:283], train_test_split = "intend", num = 6)
sol_2 = RON.try_different_method(data.iloc[177:223], train_test_split = "intend", num = 6)
sol_3 = RON.try_different_method(data.iloc[283::], train_test_split = "intend", num = 6)
sol_4 = RON.try_different_method(data.iloc[283:329], train_test_split = "intend", num = 6)
res_1 = RON.predict(data, sol_1[4], sol_1[5])
res_2 = RON.predict(data, sol_2[4], sol_2[5])
res_3 = RON.predict(data, sol_3[4], sol_3[5])
res_4 = RON.predict(data, sol_4[4], sol_4[5])
sol_5 = RON.try_different_method(data, train_test_split = "intend", num = 6)
res_5 = RON.predict(data, sol_5[4], sol_5[5])
sol_6 = RON.try_different_method(data[data.RON<110], train_test_split = "intend", num = 6)
res_6 = RON.predict(data, sol_6[4], sol_6[5])

sol_1 = RON.try_different_method(data.iloc[np.concatenate([np.arange(177, 283), np.arange(561, 564),
                                                           np.arange(582, 601)])], train_test_split = "intend", num = 6)

# alky
sol_1 = RON.try_different_method(data.iloc[:108], train_test_split = "intend", num = 6)
res_1 = RON.predict(data, sol_1[4], sol_1[5])
sol_2 = RON.try_different_method(data.iloc[np.concatenate([np.arange(0, 108),
                                                           np.arange(568, 575)])], train_test_split = "intend", num = 6)
res_2 = RON.predict(data, sol_2[4], sol_2[5])


train_x, test_x, train_y, test_y = RON.train_test_extend(data, 10)
train_x, val_x, train_y, val_y = RON.train_test_intend(data.iloc[10:-10], 4)
sol_4 = RON.try_different_method(train_x, train_y, val_x, val_y)
sol_5 = RON.try_different_method(train_x, train_y, test_x, test_y)

# cross_val
data = data.sort_values(by = "ID", ascending = False)
b = data[:43]
b1, b2 = RON.cross_val(b)

# 创建一个模型
model = GC_Model()
# 定义要搜索的超参数网格
param_grid = {'alpha_1': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
              'alpha_2': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]}
# 创建GridSearchCV对象，并传递自定义评分函数
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring="neg_mean_absolute_error")
# 执行网格搜索
train_x, test_x, train_y, test_y, train_type, test_type = RON.train_test_intend(data.iloc[177:222], 100)
grid_search.fit(train_x, train_y)
# 获取所有参数结果
results = grid_search.cv_results_
# 输出所有参数组合及其性能指标
for mean_score, params in zip(results['mean_test_score'], results['params']):
    print(f"Mean Score: {mean_score}, Parameters: {params}")
# 获取最佳参数组合
best_params = grid_search.best_params_        


"""终馏点模型"""
# =============================================================================
filename = r"E:\Syspetro\_3_汽油快评\_16_快评工作交接\8.算法对接\3.物性计算算法\训练参数_辛烷值alpha模型\T2_烷基化.xlsx"
FBP = GC_Model(physical_name = "FBP", moleculer_column = 5)
data, data_cg = FBP.data_read(filename)
sol_1 = FBP.try_different_method(data, train_test_split = "intend", num = 3)



