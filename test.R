data(Prostate, package="lasso2")
str(Prostate)

# 先把資料區分成 train=0.8, test=0.2 
set.seed(22)
train.index <- sample(x=1:nrow(Prostate), size=ceiling(0.8*nrow(Prostate) ))

train = Prostate[train.index, ]
test = Prostate[-train.index, ]

################################################################################################

# Bagging 的一開始，是要先把資料做重複抽樣，因此先將 Train 分成三個子資料集(n=33)
# Subset-1
set.seed(1)
ind_1 = sample(1:nrow(train), 33)
subset_1 = train[ind_1, ]

# Subset-1
set.seed(2)
ind_2 = sample(1:nrow(train), 33)
subset_2 = train[ind_2, ]

# Subset-3
set.seed(3)
ind_3 = sample(1:nrow(train), 33)
subset_3 = train[ind_3, ]

# 對subset_1，subset_2，subset_3分別建立線性迴歸模型，再將 Test 的預測結果平均起來
# Model-1 : linear regression
model_1 = lm(lpsa~., subset_1)
y1 = predict(model_1, test)

# Model-2 : linear regression
model_2 = lm(lpsa~., subset_2)
y2 = predict(model_2, test)

# Model-3 : linear regression
model_3 = lm(lpsa~., subset_3)
y3 = predict(model_3, test)

# Average Prediction Results
ave_y = (y1+y2+y3)/3

# MSE Comparision between three models and the bagging model
c(mean((y1 - test$lpsa)^2),   # linear regression of subset_1
  mean((y2 - test$lpsa)^2),   # linear regression of subset_2
  mean((y3 - test$lpsa)^2),   # linear regression of subset_3
  mean((ave_y - test$lpsa)^2))  # bagging model

################################################################################################

# randomForest套件可以幫我們建立隨機森林的模型
require(randomForest)

rf_model = randomForest(lpsa~.,
                        data=train,
                        ntree=150        # 決策樹的數量
)
# 預測
rf_y = predict(rf_model, test)
mean((rf_y - test$lpsa)^2) # MSE

# 這裡有一個建模中都會遇到的重要議題：「要決定多少決策樹？」(也就是ntree要設定多少？)
# 根據下圖，我們可以知道，當ntree = 70之後，整體的誤差大概就趨於穩定
# Observe that what is the best number of trees
plot(rf_model)

# 另一個參數mtry其實也需要去 Tune，這個參數代表每次抽樣時需要抽「多少個變數」的意思
# 可以使用tuneRF()來 tune mtry的值，並根據下面的結果與圖，得知每次抽樣時「抽4個變數」會是比較好的選擇
tuneRF(train[,-9], train[,9])


# 最佳化的參數後可以重寫成這樣
rf_model = randomForest(lpsa~.,
                        data = train,
                        ntree = 70,        # 決策樹的數量
                        mtry = 4
)
# 預測
rf_y = predict(rf_model, test)
mean((rf_y - test$lpsa)^2) # MSE

# 哪些變數是會對損失函數(Loss Function)最有貢獻的
# Variable Importance of Random Forest
rf_model$importance
varImpPlot(rf_model)

################################################################################################

# 使用 xgboost 會用下面這樣的SOP，提供當作參考：
# 1.將資料格式(Data.frame)，用xgb.DMatrix()轉換為 xgboost 的稀疏矩陣
# 2.設定xgb.params，也就是 xgboost 裡面的參數
# 3.使用xgb.cv()，tune 出最佳的決策樹數量(nrounds)
# 4.用xgb.train()建立模型

# 1. 將資料格式(Data.frame)，用`xgb.DMatrix()`轉換為 xgboost 的稀疏矩
require(xgboost)
dtrain = xgb.DMatrix(data = as.matrix(train[,1:8]),
                     label = train$lpsa)
dtest = xgb.DMatrix(data = as.matrix(test[,1:8]),
                    label = test$lpsa)

# 2. 設定xgb.params，也就是 xgboost 裡面的參數
xgb.params = list(
  #col的抽樣比例，越高表示每棵樹使用的col越多，會增加每棵小樹的複雜度
  colsample_bytree = 0.5,                    
  # row的抽樣比例，越高表示每棵樹使用的col越多，會增加每棵小樹的複雜度
  subsample = 0.5,                      
  booster = "gbtree",
  # 樹的最大深度，越高表示模型可以長得越深，模型複雜度越高
  max_depth = 2,           
  # boosting會增加被分錯的資料權重，而此參數是讓權重不會增加的那麼快，因此越大會讓模型愈保守
  eta = 0.03,
  # 或用'mae'也可以
  eval_metric = "rmse",                      
  objective = "reg:linear",
  # 越大，模型會越保守，相對的模型複雜度比較低
  gamma = 0)               

# 使用xgb.cv()的函式，搭配 cross validation 的技巧，找出最佳的決策樹數量nrounds
# 3. 使用xgb.cv()，tune 出最佳的決策樹數量
cv.model = xgb.cv(
  params = xgb.params, 
  data = dtrain,
  nfold = 5,     # 5-fold cv
  nrounds=200,   # 測試1-100，各個樹總數下的模型
  # 如果當nrounds < 30 時，就已經有overfitting情況發生，那表示不用繼續tune下去了，可以提早停止                
  early_stopping_rounds = 30, 
  print_every_n = 20 # 每20個單位才顯示一次結果，
) 

# 畫圖，觀察 CV 過程中Train 跟 Validation 資料的表現(紅色：Train，藍色：Validation)
tmp = cv.model$evaluation_log
# tmp = data.frame(train_rmse_mean=cv.model$train.rmse.mean,test_rmse_mean=cv.model$test.rmse.mean)

plot(x=1:nrow(tmp), y= tmp$train_rmse_mean, col='red', xlab="nround", ylab="rmse", main="Avg.Performance in CV") 
points(x=1:nrow(tmp), y= tmp$test_rmse_mean, col='blue') 
legend("topright", pch=1, col = c("red", "blue"), 
       legend = c("Train", "Validation") )

# 如果 Train 跟 Validation 相近，表示模型其實還可以訓練得更好(更複雜)
# 可以調以下參數，以提高模型複雜度的概念進行
# max_depth 調高 1 單位 (最建議調這個)
# colsample_bytree、subsample調高比例 (調這個也不錯)
# eta調低 (不得以才調這個)
# gamma 調低 (不得以才調這個)
# 如果 Train 比 Validation 好太多，就表示有 ovrfitting的問題發生，這時候上面的參數就要反過來調，以降低模型複雜度的概念來進行

# 可以利用把cv.model中的best_iteration取出來
# 自動判斷最好的 nrounds是多少，其方法是觀察 Train 跟 Validation 之間的表現差異
# 獲得 best nround
best.nrounds = cv.model$best_iteration 
best.nrounds

# 4. 用xgb.train()建立模型
xgb.model = xgb.train(paras = xgb.params, 
                      data = dtrain,
                      nrounds = best.nrounds) 

# 如果要畫出 xgb 內的所有決策樹，可以用以下函式(但因為會很多，這裡就不畫了)
# library("DiagrammeR")
# xgb.plot.tree(model = xgb.model)

# 預測
xgb_y = predict(xgb.model, dtest)
mean((xgb_y - test$lpsa)^2) # MSE

################################################################################################

# R Code for Stacking Implementg
# 第一階段(Stacking)
# 一開始，我們把訓練資料 Train 分成三份(3-folds)
# 3-folds
n = 3
n.folds = rep(1:n, each=nrow(train)/n)
train.folds = split(train, n.folds)

# 用 linear regression 為例，撰寫一個 stacking 架構，並且儲存meta-x 跟 meta-y的值
meta.x = vector()
meta.y = list()

# 1st fold for validation
stacking.train = rbind(train.folds[[2]], train.folds[[3]])
stacking.valid = train.folds[[1]]
stacking.test = test

model_1 = lm(lpsa~., stacking.train)

tmp.meta.x = predict(model_1, stacking.valid)
tmp.meta.y = predict(model_1, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[1]] = tmp.meta.y

# 2nd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[3]])
stacking.valid = train.folds[[2]]
stacking.test = test

model_1 = lm(lpsa~., stacking.train)

tmp.meta.x = predict(model_1, stacking.valid)
tmp.meta.y = predict(model_1, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[2]] = tmp.meta.y

# 3rd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[2]])
stacking.valid = train.folds[[3]]
stacking.test = test

model_1 = lm(lpsa~., stacking.train)

tmp.meta.x = predict(model_1, stacking.valid)
tmp.meta.y = predict(model_1, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[3]] = tmp.meta.y

# 第一個線性模型跑完三次之後，就會得到一組meta.x跟meta.y
# 再跟本來的實際值(train$lpsa, test$lpsa)結合，成為第一個模型輸出的 meta.train.1跟 meta.test.1
# Average Meta.X of Test
mean.meta.y = (meta.y[[1]] + meta.y[[2]] + meta.y[[3]]) / 3

meta.train.1 = data.frame('meta.x' = meta.x, 
                          y=train$lpsa)

meta.test.1 = data.frame('mete.y' = mean.meta.y, 
                         y = test$lpsa)

# 接下來，第二模型用randomForest，也是一樣的流程
meta.x = vector()
meta.y = list()

# 1st fold for validation
stacking.train = rbind(train.folds[[2]], train.folds[[3]])
stacking.valid = train.folds[[1]]
stacking.test = test

model_2 = randomForest(lpsa~.,data = stacking.train,
                        ntree = 70,        # 決策樹的數量
                        mtry = 4)

tmp.meta.x = predict(model_2, stacking.valid)
tmp.meta.y = predict(model_2, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[1]] = tmp.meta.y

# 2nd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[3]])
stacking.valid = train.folds[[2]]
stacking.test = test

model_2 = randomForest(lpsa~.,data = stacking.train,
                       ntree = 70,        # 決策樹的數量
                       mtry = 4)

tmp.meta.x = predict(model_2, stacking.valid)
tmp.meta.y = predict(model_2, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[2]] = tmp.meta.y

# 3rd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[2]])
stacking.valid = train.folds[[3]]
stacking.test = test

model_2 = randomForest(lpsa~.,data = stacking.train,
                       ntree = 70,        # 決策樹的數量
                       mtry = 4)

tmp.meta.x = predict(model_2, stacking.valid)
tmp.meta.y = predict(model_2, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[3]] = tmp.meta.y

# Average Meta.X of Test
mean.meta.y = (meta.y[[1]] + meta.y[[2]] + meta.y[[3]]) / 3

meta.train.2 = data.frame('meta.x' = meta.x, 
                          y=train$lpsa)

meta.test.2 = data.frame('mete.y' = mean.meta.y, 
                         y = test$lpsa)


# 仿照上面的流程，第三個模型則用 CART 決策樹
require(rpart)
meta.x = vector()
meta.y = list()

# 1st fold for validation
stacking.train = rbind(train.folds[[2]], train.folds[[3]])
stacking.valid = train.folds[[1]]
stacking.test = test

model_3 = rpart(lpsa~., stacking.train)

tmp.meta.x = predict(model_3, stacking.valid)
tmp.meta.y = predict(model_3, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[1]] = tmp.meta.y

# 2nd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[3]])
stacking.valid = train.folds[[2]]
stacking.test = test

model_3 = rpart(lpsa~., stacking.train)

tmp.meta.x = predict(model_3, stacking.valid)
tmp.meta.y = predict(model_3, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[2]] = tmp.meta.y

# 3rd fold for validation
stacking.train = rbind(train.folds[[1]], train.folds[[2]])
stacking.valid = train.folds[[3]]
stacking.test = test

model_3 = rpart(lpsa~., stacking.train)

tmp.meta.x = predict(model_3, stacking.valid)
tmp.meta.y = predict(model_3, stacking.test)

meta.x = c(meta.x, tmp.meta.x)
meta.y[[3]] = tmp.meta.y

# Average Meta.X of Test
mean.meta.y = (meta.y[[1]] + meta.y[[2]] + meta.y[[3]]) / 3

meta.train.3 = data.frame(`meta.x` = meta.x, 
                          y=train$lpsa)

meta.test.3 = data.frame(`mete.y` = mean.meta.y, 
                         y = test$lpsa)


# 第二階段(Blending)
# 手中有三組 Meta-Train 跟 三組 Meta-Test
c(dim(meta.train.1), dim(meta.test.1))

# 要建構第二階段的 Meta-Model，我這裡拿xgboost的模型來做

### Meta- Model Construction 
# 先把三個 Meta-Train合併一起：
big.meta.train = rbind(meta.train.1, meta.train.2, meta.train.3)

# 轉換成 xgboost 的格式
dtrain = xgb.DMatrix(data = as.matrix(big.meta.train[,1]), label = big.meta.train[, 2])

# 訓練 XGboost 模型
# 其中xgb.params 直接拿第二節(Boosting)的設定
# 簡單用 nrounds = 100
xgb.model = xgb.train(paras = xgb.params, data = dtrain, nrounds = 100) 

# 對三組 Meta-Test進行預測：
dtest.1 = xgb.DMatrix(data = as.matrix(meta.test.1[,1]), label = meta.test.1[, 2])
final_1 = predict(xgb.model, dtest.1)

dtest.2 = xgb.DMatrix(data = as.matrix(meta.test.2[,1]), label = meta.test.2[, 2])
final_2 = predict(xgb.model, dtest.2)

dtest.3 = xgb.DMatrix(data = as.matrix(meta.test.3[,1]), label = meta.test.3[, 2])
final_3 = predict(xgb.model, dtest.3)

# 把三組結果平均起來，然後算 MSE
final_y = (final_1 + final_2 + final_3)/3
mean((final_y - test$lpsa)^2) # MSE




