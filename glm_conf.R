#Package
require(caret)
require(glmnet)

#Chargement
load("dta.RData")
dta <- dta[ , -c(1,10, 11,12,13,14)]

#Train-Test
set.seed(123)
train_indices <- sample(1:nrow(dta), 2148)
train <- dta[train_indices,] 
test <- dta[-train_indices, ] 

train_control <- trainControl(method = "LGOCV", number = 10, p=0.7) 

#GLM 
glm_model <- train(Depression_level ~ ., data = train, 
                   method = "glm", 
                   family = gaussian, 
                   trControl = train_control)
glm_model
predictions_glm <- predict(glm_model, newdata = test)
RMSE_glm <- sqrt(mean((predictions_glm - test$Depression_level)^2)) 

#Mixte (lasso-ridge)
lambda <- seq(0.001, 2, length = 20)
alpha <- seq (0, 1, length=10)
mixte_model <- train(Depression_level ~ ., data = train, 
                     method = "glmnet", 
                     trControl = train_control,
                     tuneGrid = expand.grid(alpha = alpha,  
                                            lambda = lambda))  



mixte_model$bestTune
predictions_mixte <- predict(mixte_model, newdata = test)
RMSE_mixte <- sqrt(mean((predictions_mixte - test$Depression_level)^2)) 
res <- mixte_model$results
plot(res[,2], res[,3])

