rm(list = ls())

if (!require("ggplot2")) install.packages("ggplot2")
library('ggplot2')
if (!require("caret")) install.packages("caret")
library('caret')
if (!require("doParallel")) install.packages("doParallel")
library('doParallel')
if (!require("MLmetrics")) install.packages("MLmetrics")
library('MLmetrics')

set.seed(100)
path = getwd()
# path = '~/ML' # 自定義路徑
dir.create(file.path(path, "model"), showWarnings = FALSE) # model儲存位置
gc()

#####################################################################
#####################################################################
set.seed(100)
dataset<-read.csv(paste0(path,"/UCI_Credit_Card.csv"), header=TRUE)
sum(is.na(dataset))
dataset[2:24]
dataset[2:24] <- prodNA(dataset[2:24], noNA = 0.01)
head(dataset)
set.seed(100)
train.index <- sample(x=1:nrow(dataset), size=ceiling(0.8*nrow(dataset) ))
train = dataset[train.index, ]
test = dataset[-train.index, ]
write.csv(train,paste0(path,"/train.csv"),row.names = FALSE)
write.csv(test,paste0(path,"/test.csv"),row.names = FALSE)
#####################################################################
#####################################################################

TrainDataFile = 'train.csv'
TestDataFile  = 'test.csv'
KeyColumn  = 'ID'
TargetColumn  = 'default.payment.next.month'
OutputAnswer  = 'outputdata'
MissingTypes = c("", "NA")

TrainData = read.csv(paste0(path,'/',TrainDataFile), header = T, sep=",", na.strings = MissingTypes)
TestData = read.csv(paste0(path,'/',TestDataFile), header = T, sep=",", na.strings = MissingTypes)

numeric_col = c('') #names(TrainData[12:23])
integer_col = c('')
factor_col = c('SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','default.payment.next.month')
Date_col = c('')
character_col = c('ID') #c('ID','POLICY_NO','APPLY_NO','EMAIL','PHONE')

typechange = function(dataset,KeyColumn,numeric_col,integer_col,factor_col,Date_col,character_col){
  dataset = as.data.frame(dataset)
  if(sum(numeric_col!="")>0) dataset[numeric_col] <- lapply(dataset[numeric_col], as.numeric)
  if(sum(integer_col!="")>0) dataset[integer_col] <- lapply(dataset[integer_col], as.integer)
  if(sum(factor_col!="")>0) dataset[factor_col] <- lapply(dataset[factor_col], as.factor)
  Unikey = dataset[[KeyColumn]]
  dataset = dataset[,-which(colnames(dataset) %in% c(Date_col,character_col))]
  dataset = as.data.frame(cbind(Unikey,dataset))
  return(dataset)
}

TrainData = typechange(TrainData,KeyColumn,numeric_col,integer_col,factor_col,Date_col,character_col)
TestData = typechange(TestData,KeyColumn,numeric_col,integer_col,factor_col,Date_col,character_col)

Y_ind = TrainData[[TargetColumn]] == '' | is.na(TrainData[[TargetColumn]])
print(paste("應變數異常筆數為",sum(Y_ind)))

TrainData[Y_ind,'Unikey'] #顯示有異常的資料key


clearNA = function(dataset){
  dataset = as.data.frame(dataset)
  ind_col = apply(dataset, 2,function(x) all(is.na(x)|x==''))
  if(sum(ind_col)>0) dataset = dataset[,-which(ind_col)]
  return(dataset)
}

TrainData = clearNA(TrainData)

apply(TrainData[,-1], 2,function(x) sum(is.na(x)))


formula = paste(get('TargetColumn')," ~ .")
dummies = dummyVars(formula, data = TrainData[-1]) # 轉dummy函數
TrainData_dmy = predict(dummies, TrainData) # 建立訓練資料的虛擬變量
TestData_dmy = predict(dummies, TestData) # 建立訓練資料的虛擬變量
# dim(TrainData_dmy)
# dim(TestData_dmy)
head(TrainData_dmy[,1:5])

# 建立多核運算環境
cl<-makeCluster(3) #建議不要超過4，自行根據硬體設備設定
registerDoParallel(cl)
#stopCluster(cl) #不使用時可以用此語法關閉多核運行

# 建立n個fold進行cross validation 以及模型的Control設定
# 如果針對演算法要詳細設定，可以各別建立Control函數
fitControl <- trainControl(## 3-fold CV
  method = "cv",
  number = 3 ) 

# 訓練資料及測試資料去除識別欄位(順便備份)
training = TrainData[-1]
testing = TestData[-1]

# 模型表現函數
report_metrics = function(method,y_ture,y_pred,y_score){
  ans = data.frame(Algorithm = method,
                   ROC_AUC = AUC(y_pred, y_ture),
                   Accuracy = Accuracy(y_pred, y_ture),
                   Precision = Precision(y_ture, y_pred, positive = '1'),
                   Recall = Recall(y_ture, y_pred, positive = '1'),
                   F1_Score = F1_Score(y_ture, y_pred, positive  = '1')
  )
  return(ans)
}



################################################################

# 建立多核運算環境
cl<-makeCluster(3) #建議不要超過4，自行根據硬體設備設定
registerDoParallel(cl)
#stopCluster(cl) #不使用時可以用此語法關閉多核運行

# 建立n個fold進行cross validation 以及模型的Control設定
# 如果針對演算法要詳細設定，可以各別建立Control函數
fitControl <- trainControl(## 3-fold CV
  method = "cv",
  number = 3 )

# 訓練資料及測試資料去除識別欄位訓練(備份用意)
training = TrainData[-1]
testing = TestData[-1]

# 模型表現函數
report_metrics = function(method,y_ture,y_pred,y_score){
  ans = data.frame(Algorithm = method,
                   ROC_AUC = AUC(y_pred, y_ture),
                   Accuracy = Accuracy(y_pred, y_ture),
                   Precision = Precision(y_ture, y_pred, positive = '1'),
                   Recall = Recall(y_ture, y_pred, positive = '1'),
                   F1_Score = F1_Score(y_ture, y_pred, positive  = '1')
  )
  return(ans)
}

# GLM模型訓練
set.seed(200)
formula = paste(get('TargetColumn')," ~ ",paste(names(training), collapse= "+"))
formula=as.formula(formula)
glmFit <- train(formula, data = training,
                method = "glm",
                trControl = fitControl)
glmFit

# 訓練資料模型表現
glm_train_pred = predict(glmFit, training)
glm_train_prob = predict(glmFit, training, type = "prob")
glm_train_report = report_metrics(glmFit$method,training$default.payment.next.month,glm_train_pred,glm_train_prob)
glm_train_report

# 模型重要變數
glm_imp = varImp(glmFit, scale = TRUE)
plot(glm_imp)

# 訓練資料模型表現
glm_test_pred = predict(glmFit, testing)
glm_test_prob = predict(glmFit, testing, type = "prob")
glm_test_report = report_metrics(glmFit$method,testing$default.payment.next.month,glm_test_pred,glm_test_prob)
glm_test_report







################################################################

# ADAB模型訓練
library(fastAdaboost)
set.seed(200)
formula = paste(get('TargetColumn')," ~ .")
formula=as.formula(formula)
adabFit <- train(formula, data = training, 
                 method = "adaboost", 
                 trControl = fitControl)
save(adabFit, file = "./model/adabFit.rda") # 儲存模型檔
adabFit


# 訓練資料模型表現
adab_train_pred = predict(adabFit, training)
adab_train_prob = predict(adabFit, training, type = "prob")
adab_train_report = report_metrics(adabFit$method,training$default.payment.next.month,adab_train_pred,adab_train_prob)
adab_train_report

# 模型重要變數
adab_imp = varImp(adabFit, scale = TRUE)
plot(adab_imp)

# 測試資料模型表現
adab_test_pred = predict(adabFit, testing)
adab_test_prob = predict(adabFit, testing, type = "prob")
adab_test_report = report_metrics(adabFit$method,testing$default.payment.next.month,adab_test_pred,adab_test_prob)
adab_test_report


TrainData_dmy = predict(dummies, TrainData) # 建立訓練資料的虛擬變量
TestData_dmy = predict(dummies, TestData) # 建立訓練資料的虛擬變量

formula = paste(get('TargetColumn')," ~ .")
formula=as.formula(formula)
xgbFit2 <- train(y=training$default.payment.next.month, x = TrainData_dmy, 
                method = "xgbTree", 
                trControl = fitControl)

xgbFit2

xgb_test2_pred = predict(xgbFit2, TestData_dmy)
xgb_test2_prob = predict(xgbFit2, TestData_dmy, type = "prob")
xgb_test2_report = report_metrics(xgbFit2$method,testing$default.payment.next.month,xgb_test2_pred,xgb_test2_prob)
xgb_test2_report
xgb_test_report

table(xgb_test2_pred)

xgb_train2_pred = predict(xgbFit2, training)
xgb_train2_prob = predict(xgbFit2, training, type = "prob")
xgb_train2_report = report_metrics(xgbFit2$method,training$default.payment.next.month,xgb_train2_pred,xgb_train2_prob)
xgb_train2_report



formula = paste(get('TargetColumn')," ~ .")
formula=as.formula(formula)
glmFit2 <- train(formula, data = training, 
                 method = "glm", 
                 family='binomial',
                 trControl = fitControl)
glmFit2

# 測試資料模型表現
glm_test2_pred = predict(glmFit2, testing)
glm_test2_prob = predict(glmFit2, testing, type = "prob")
glm_test2_report = report_metrics(glmFit2$method,testing[[TargetColumn]],glm_test2_pred,glm_test2_prob)
glm_test2_report
glm_test_report

glmFit <- train(formula, data = training, 
                method = "glm", 
                trControl = fitControl)
glmFit
glm_test_pred = predict(glmFit, testing)
glm_test_prob = predict(glmFit, testing, type = "prob")
glm_test_report = report_metrics(glmFit$method,testing[[TargetColumn]],glm_test_pred,glm_test_prob)
glm_test_report



require(caret)
require(devtools)
library(lightgbm)

model <-caretModel.LGBM()



































formula = paste(get('TargetColumn')," ~ .")
formula=as.formula(formula)
glmFit <- train(formula, data = training, 
                method = "glm", 
                trControl = fitControl)
save(glmFit, file = "./model/glmFit.rda") # 儲存模型檔
glmFit























imbal_train <- twoClassSim(10000, intercept = -20, linearVars = 20)
imbal_test  <- twoClassSim(10000, intercept = -20, linearVars = 20)
table(imbal_train$Class)


# 原本樣本分佈
table(TrainData[[TargetColumn]])

# 若要使用請將down_train自行改為TrainData
down_train <- downSample(x = TrainData[,-which(colnames(dataset)==TargetColumn)],
                         y = TrainData[[TargetColumn]])
names(down_train)[names(down_train)=="Class"]=get('TargetColumn')
table(down_train[[TargetColumn]]) 

# 若要使用請將up_train自行改為TrainData
up_train <- upSample(x = TrainData[,-which(colnames(dataset)==TargetColumn)],
                     y = TrainData[[TargetColumn]])  
names(up_train)[names(up_train)=="Class"]=get('TargetColumn')
table(up_train[[TargetColumn]]) 

# 若要使用請將over_train自行改為TrainData
formula = as.formula(paste(get('TargetColumn')," ~ ."))
over_train <- ovun.sample(formula, data = TrainData, method = "over")$data
table(over_train[[TargetColumn]]) 

# 若要使用請將under_train自行改為TrainData
formula = as.formula(paste(get('TargetColumn')," ~ ."))
under_train <- ovun.sample(formula, data = TrainData, method = "under")$data
table(under_train[[TargetColumn]]) 


# 若要使用請將smote_train自行改為TrainData
library(DMwR)
formula = as.formula(paste(get('TargetColumn')," ~ ."))
smote_train <- SMOTE(formula, data  = TrainData)                         
table(smote_train[[TargetColumn]]) 

# 若要使用請將smote_train自行改為TrainData
library(ROSE)
formula = as.formula(paste(get('TargetColumn')," ~ ."))
rose_train <- ROSE(formula, data  = TrainData)$data                         
table(rose_train[[TargetColumn]]) 




unique(c('5','7','5','5','7'))
length(unique(down_train$Unikey))




library(DMwR)
set.seed(9560)
smote_train <- SMOTE(Class ~ ., data  = imbal_train)                         
table(smote_train$Class) 



dir.create(file.path(path, "model"), showWarnings = FALSE)

data(mdrr)
nzv <- nearZeroVar(mdrrDescr)
filteredDescr <- mdrrDescr[, -nzv]
dim(filteredDescr)
names(mdrrDescr)[nzv]


ltfrDesign <- matrix(0, nrow=6, ncol=6)
ltfrDesign[,1] <- c(1, 1, 1, 1, 1, 1)
ltfrDesign[,2] <- c(1, 1, 1, 0, 0, 0)
ltfrDesign[,3] <- c(0, 0, 0, 1, 1, 1)
ltfrDesign[,4] <- c(1, 0, 0, 1, 0, 0)
ltfrDesign[,5] <- c(0, 1, 0, 0, 1, 0)
ltfrDesign[,6] <- c(0, 0, 1, 0, 0, 1)
ltfrDesign=as.data.frame(ltfrDesign)
comboInfo <- findLinearCombos(ltfrDesign)
names(ltfrDesign)[comboInfo$remove]

library('caret')
library('gbm')
library(doParallel)
cl<-makeCluster(2)
registerDoParallel(cl)
stopImplicitCluster()
getDoParWorkers()

stopCluster(cl)


# train control
fitControl <- trainControl(## 10-fold CV
  method = "cv",
  number = 3)

# train by gbm
train_tmp = TrainData[-1]

set.seed(100)
gbmFit1 <- train(default.payment.next.month ~ ., data = train_tmp,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
gbmFit1

# grid
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9),
                        n.trees = (1:30)*50,
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

nrow(gbmGrid)
set.seed(100)
gbmFit2 <- train(default.payment.next.month ~ ., data = train_tmp,
                 method = "gbm",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE,
                 tuneGrid = gbmGrid)
gbmFit2

# plot
trellis.par.set(caretTheme())
plot(gbmFit2)
plot(gbmFit2, metric = "Kappa")
plot(gbmFit2,  plotType = "level",
     scales = list(x = list(rot = 90)))


whichTwoPct <- tolerance(gbmFit2$results, metric = "ROC",
                         tol = 2, maximize = TRUE)


gbmFit2$resample



# pred
# train by gbm
test_tmp = TestData[2:24]
y=predict(gbmFit2, newdata = TestData)
x=predict(gbmFit2, test_tmp,type = "prob")


densityplot(gbmFit2, pch = "|")


confusion_matrix = table(x,TestData$default.payment.next.month,dnn = c("預測", "實際"))
accuracy = sum(diag(confusion_matrix)) / sum(confusion_matrix)
LH=diag(confusion_matrix)/colSums(confusion_matrix)
confusion_matrix
accuracy
LH

library (ROCR)
performance(y, "prec", "rec")
posPredValue(y, TestData$default.payment.next.month, positive="0")
sensitivity(y, TestData$default.payment.next.month, positive="0")

stopImplicitCluster()



x=gbmFit2$results
which(x$Accuracy==0.820125)

gbmFit2$
varImp(gbmFit2,numTrees = 100,interaction.depth=5,shrinkage=0.1,minobsinnode=20)

gbmFit2$bestTune
gbmFit2$modelInfo

varImp(gbmFit)

a=varImp(gbmFit1)

gbmFit2$modelInfo

a$calledFrom

table(TestData$default.payment.next.month)

dummies <- dummyVars(default.payment.next.month ~ ., data = TrainData)
new_tmp<-(predict(dummies, newdata = TrainData))

new_tmp=as.data.frame(new_tmp)


table(mdrrDescr$nR11)

nzv <- nearZeroVar(mdrrDescr, saveMetrics= TRUE)

nzv[nzv$nzv,][1:10,]
dim(mdrrDescr)
nzv <- nearZeroVar(mdrrDescr)
filteredDescr <- mdrrDescr[, -nzv]
dim(filteredDescr)


descrCor <-  cor(filteredDescr)
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .999)

summary(descrCor[upper.tri(descrCor)])
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
filteredDescr <- filteredDescr[,-highlyCorDescr]
descrCor2 <- cor(filteredDescr)
summary(descrCor2[upper.tri(descrCor2)])


mdrrDescr$u=c(rep(5,500),rep(10,500),c())
table(u)
nearZeroVar(u)

nrow(mdrrDescr)
mdrrDescr$u=c(rep(5,250),rep(10,250),rep(1,28))


get('TargetColumn')
TrainData_dmy = dummyVars(default.payment.next.month ~ ., data = TrainData[-1])
TrainData_dmy = dummyVars(formula, data = TrainData[-1])
TrainData_dmy = dummyVars(get('TargetColumn') ~ ., data = TrainData[-1])

get('TargetColumn')

formula = paste(get('TargetColumn')," ~ .")
dummies = dummyVars(formula, data = TrainData[-1]) # 轉dummy函數
TrainData_dmy = predict(dummies, TrainData) # 建立訓練資料的虛擬變量
TestData_dmy = predict(dummies, TestData) # 建立訓練資料的虛擬變量
# dim(TrainData_dmy)
# dim(TestData_dmy)


nzv = nearZeroVar(TrainData)
if(length(nzv)>0) TrainData <- TrainData[, -nzv]

comboInfo <- findLinearCombos(TrainData)
comboInfo

comboInfo = findLinearCombos(TrainData) # 找出線性相關欄位
a = TrainData[, -comboInfo$remove] # 排除相關性欄位



t <- preProcess(TrainData, method = c("center", "scale"))

trainTransformed <- predict(t, TrainData)
testTransformed <- predict(t, TestData)

head(trainTransformed)
head(TrainData)

trainTransformed

preProcValues = preProcess(TrainData, method = c("center", "scale"))
trainTransformed = predict(preProcValues, TrainData) # 訓練資料轉換
testTransformed = predict(preProcValues, TestData) # 測試資料轉換


TrainData = data.frame(TrainData$Unikey,trainTransformed[-1])
TestData = data.frame(TestData$Unikey,testTransformed[-1])
head(TrainData_N)


formula = paste(get('TargetColumn')," ~ ",paste(names(training), collapse= "+"))
glmFit <- train(default.payment.next.month  ~  LIMIT_BAL+SEX+EDUCATION+MARRIAGE+AGE+PAY_0+PAY_2+PAY_3+PAY_4+PAY_5+PAY_6+BILL_AMT1+BILL_AMT2+BILL_AMT3+BILL_AMT4+BILL_AMT5+BILL_AMT6+PAY_AMT1+PAY_AMT2+PAY_AMT3+PAY_AMT4+PAY_AMT5+PAY_AMT6+default.payment.next.month, data = training,
                method = "glm",
                trControl = fitControl)

glmFit2 <- train(formula, data = training,
                method = "glm",
                trControl = fitControl)


glmFit$bestTune

formula=as.formula(formula)

character()

names(training)
split(names(training),',')

glmFit$resample
glmFit$terms

glmFit$resample
glmFit$results$Accuracy

class(glmFit)
glmFit$method
glmFit$perfNames
glmFit$metric


report_fits = function(modelfit){
  list(Algorithm = modelfit$method,
       fold_AUC = 0.25
  )
}

report_metrics = function(method,y_ture,y_pred,y_score){
  ans = data.frame(Algorithm = method,
                   ROC_AUC = AUC(y_pred, y_ture),
                   Accuracy = Accuracy(y_pred, y_ture),
                   Precision = Precision(y_ture, y_pred, positive = '1'),
                   Recall = Recall(y_ture, y_pred, positive = '1'),
                   F1_Score = F1_Score(y_ture, y_pred, positive  = '1')
  )
 return(ans)
}


library('MLmetrics')
y_ture = training$default.payment.next.month
y_pred = glm_pred
y_score = predict(glmFit, testing, type = "prob")[,2]




xTab <- table(y_pred, y_ture)
clss <- as.character(sort(unique(y_pred)))
r <- matrix(NA, ncol = 7, nrow = 1,
            dimnames = list(c(),c('Acc',
                                  paste("P",clss[1],sep='_'),
                                  paste("R",clss[1],sep='_'),
                                  paste("F",clss[1],sep='_'),
                                  paste("P",clss[2],sep='_'),
                                  paste("R",clss[2],sep='_'),
                                  paste("F",clss[2],sep='_'))))
r[1,1] <- sum(xTab[1,1],xTab[2,2])/sum(xTab) # Accuracy
r[1,2] <- xTab[1,1]/sum(xTab[,1]) # Miss Precision
r[1,3] <- xTab[1,1]/sum(xTab[1,]) # Miss Recall
r[1,4] <- (2*r[1,2]*r[1,3])/sum(r[1,2],r[1,3]) # Miss F
r[1,5] <- xTab[2,2]/sum(xTab[,2]) # Hit Precision
r[1,6] <- xTab[2,2]/sum(xTab[2,]) # Hit Recall
r[1,7] <- (2*r[1,5]*r[1,6])/sum(r[1,5],r[1,6]) # Hit F
r





glm_pred = predict(glmFit, training)
glm_pred = predict(glmFit, testing)


glm_prob = predict(glmFit, testing, type = "prob")





varImp(glmFit, scale=TRUE)
varImp(glmFit)

importance(glmFit)

subset(training[])

library('rJava')
library('extraTrees')
set.seed(200)
formula = paste(get('TargetColumn')," ~ ",paste(names(training), collapse= "+"))
formula=as.formula(formula)

y = get('TargetColumn')
x = setdiff(names(training), y)
extFit <- train(x=training[x],
                y=training[[y]],
                method = "extraTrees",
                trControl = fitControl)
save(extFit, file = "./model/extFit.rda") # 儲存模型檔
extFit

training[x]


set.seed(200)
formula = paste(get('TargetColumn')," ~ ",paste(names(training), collapse= "+"))
formula=as.formula(formula)
rfFit <- train(formula, data = training,
               method = "rf",
               trControl = fitControl)
save(rfFit, file = "./model/rfFit.rda") # 儲存模型檔
rfFit

rf_train_pred = predict(rfFit, training)
rf_train_prob = predict(rfFit, training, type = "prob")
rf_train_report = report_metrics(rfFit$method,training$default.payment.next.month,rf_train_pred,rf_train_prob)
rf_train_report

rf_train_pred
rf_train_prob
rfFit$method

length(rf_train_pred)
length(training$default.payment.nrf.month)


set.seed(200)
formula = paste(get('TargetColumn')," ~ ",paste(names(training), collapse= "+"))
formula=as.formula(formula)
gbmFit <- train(formula, data = training,
                method = "gbm",
                trControl = fitControl)
save(gbmFit, file = "./model/gbmFit.rda") # 儲存模型檔
gbmFit
