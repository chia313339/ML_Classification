# .rs.restartR()
rm(list = ls())
path = getwd()
# path = '~/ML/ML_Classification' # 自定義路徑
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


################################################################





















