rm(list = ls())
# path = getwd()
path = '~/ML/ML_Classification' # 自定義路徑
gc()

dataset<-read.csv(paste0(path,"/UCI_Credit_Card.csv"), header=TRUE)

set.seed(100)
train.index <- sample(x=1:nrow(dataset), size=ceiling(0.8*nrow(dataset) ))

train = dataset[train.index, ]
test = dataset[-train.index, ]

write.csv(train,paste0(path,"/train.csv"),row.names = FALSE)
write.csv(test,paste0(path,"/test.csv"),row.names = FALSE)

TrainDataFile = 'train.csv'
TestDataFile  = 'test.csv'
TargetColumn  = 'default.payment.next.month'
DropColumns   = c('ID') #c('ID','POLICY_NO','APPLY_NO','EMAIL','PHONE')
OutputAnswer  = 'outputdata'





