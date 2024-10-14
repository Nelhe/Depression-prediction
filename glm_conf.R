#Package
require(caret)
require(glmnet)

#Chargement
load("dta.RData")
dta <- dta[ , -c(1,10, 11,12,13,14)]

for (i in c(1, 3:7, 9:22)) contrasts(dta[,i]) <- contr.sum(nlevels(dta[,i]))

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
glm_model$finalModel$coefficients
car::Anova(glm_model$finalModel, type = 'III')
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

dta_err <- data.frame(Valeur = c(test$Depression_level, predictions_glm),
                      Type = rep(c("Observé", "Prédit"), each = 537))
ggplot(dta_err) + aes(y = Valeur, x = Type) + 
  geom_boxplot()

summary(test$Depression_level)
summary(predictions_glm)

# Seuil d'alerte

library(pROC)

obs <- ifelse(test$Depression_level > 9, 1, 0)
seuils <- seq(0.5, 8, by = 0.1)

aucs <- rep(0, length(seuils))

for (i in 1:length(seuils)){
  pred <- ifelse(predictions_glm > seuils[i], 1, 0)
  roc_obj <- roc(obs, pred)
  aucs[i] <- auc(roc_obj)
}

plot(seuils, aucs)

s <- seuils[which(aucs == max(aucs))]

table(test$Depression_level > 9, predictions_glm > s)
