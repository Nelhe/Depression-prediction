#Package
require(caret)

#Chargement
load("dta.RData")
dta <- dta[ , -c(1,10, 11,12,13,14)]

#Train-Test
set.seed(123)
train_indices <- sample(1:nrow(dta), 2148)
train <- dta[train_indices,] 
test <- dta[-train_indices, ] 

train_control <- trainControl(method = "LGOCV", number = 10, p=0.7) 

#Modele 
glm_model <- train(Depression_level ~ ., data = train, 
                   method = "glm", 
                   family = gaussian, 
                   trControl = train_control)
glm_model
predictions <- predict(glm_model, newdata = test)
