
################################
# Visualisation des données
################################

library(ggplot2)

load("dta.RData")
dta <- dta[, -c(1, 9:11, 13:14)] |>
  dplyr::mutate(Depression_severity = as.factor(ifelse(Depression_severity == 0, 0, 1)))

summary(dta)

## Visualisation de la variable à prédire
ggplot(dta) + aes(x = Depression_severity) +
  geom_bar(alpha = 0.8, fill = 'cornsilk', color = 'gray10') +
  labs(title = "Répartition des dépressions", y = NULL) +
  theme_minimal()


################################
# Entraînement des algorithmes
################################

set.seed(123)

ind_train <- sample(1:nrow(dta), 0.8*nrow(dta))   # 80/20

dta_train <- dta[ind_train,]
dta_test <- dta[-ind_train,]

segs <- pls::cvsegments(nrow(dta_train), 10)

source("Func_perf.R")


# k plus proches voisins
library(kknn)

tune_knn <- expand.grid(k = 1:50,
                        s = seq(0.2, 0.5, by = 0.05))

k_opti <- rep(0, 10)
seuil_opti <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]
  
  f1_scores <- rep(0, nrow(tune_knn))
  
  for (i in 1:nrow(tune_knn)){
    mod_knn <- kknn(Depression_severity ~., train = train, test = valid, k = tune_knn[i,1])
    
    pred_classe <- apply(mod_knn$prob, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                         s = tune_knn[i,2])
    
    f1_scores[i] <- f1(as.numeric(valid$Depression_severity) - 1, 
                       as.numeric(pred_classe))
  }
  
  k_opti[[k]] <- tune_knn[which(f1_scores == max(f1_scores))[1],1]
  seuil_opti[[k]] <- tune_knn[which(f1_scores == max(f1_scores))[1],2]
  
  print(paste("kNN fold :", k, "/ 10 - DONE"))
}

k_opti
seuil_opti

k_knn <- sort(table(k_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()
s_knn <- sort(table(seuil_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()


mod_knn_opti <- kknn::kknn(Depression_severity ~., train = dta_train, test = dta_test, k = k_knn)
pred_classe_knn <- apply(mod_knn_opti$prob, MARGIN = 1, FUN = function(x) return(ifelse(x[2] > s_knn, 1, 0)))

conf_knn <- table(dta_test$Depression_severity, pred_classe_knn)
conf_knn


# GLM
tune_glm <- expand.grid(s = seq(0.05, 0.5, by = 0.05))

seuil_opti <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]

  f1_scores <- rep(0, nrow(tune_glm))
  
  for (i in 1:nrow(tune_glm)){
    
    mod_glm <- glm(Depression_severity ~., data = train, family = 'binomial')
    pred <- predict(mod_glm, newdata = valid, type = 'response')
    pred_classe <- ifelse(pred >= tune_glm[i,1], 1, 0)

    f1_scores[i] <- f1(as.numeric(valid$Depression_severity) - 1, 
                       as.numeric(pred_classe))
  }
  
  seuil_opti[[k]] <- tune_glm[which(f1_scores == max(f1_scores))[1],1]
  print(paste("GLM fold :", k, "/ 10 - DONE"))
}

seuil_opti

s_glm <- sort(table(seuil_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()

mod_glm_opti <- glm(Depression_severity ~., data = dta_train, family = 'binomial')
pred_glm <- predict(mod_glm_opti, newdata = dta_test, type = 'response')
pred_classe_glm <- ifelse(pred_glm >= s_glm, 1, 0)

conf_glm <- table(dta_test$Depression_severity, pred_classe_glm)
conf_glm

# GLM net
library(glmnet)

tune_glmnet <- expand.grid(alp = seq(0, 1, length = 5),
                           lb = seq(1e-4, 1e-1, length = 5),
                           s = seq(0.05, 0.5, by = 0.05))

alpha_opti <- rep(0, 10)
lambda_opti <- rep(0, 10)
seuil_opti <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]
  
  f1_scores <- rep(0, nrow(tune_glmnet))
  
  for (i in 1:nrow(tune_glmnet)){
    mod_glmnet <- glmnet(x = as.matrix(train[-8]), y = train$Depression_severity, 
                         family = 'binomial',
                         alpha = tune_glmnet[i,1], lambda = tune_glmnet[i,2])
    
    pred <- predict(mod_glmnet, newx = as.matrix(valid[-8]), type = 'response')
    pred_classe <- ifelse(pred > tune_glmnet[i,3], 1, 0)
    
    f1_scores[i] <- f1(as.numeric(valid$Depression_severity) - 1, 
                       as.numeric(pred_classe))
  }
  
  alpha_opti[[k]] <- tune_glmnet[which(f1_scores == max(f1_scores))[1],1]
  lambda_opti[[k]] <- tune_glmnet[which(f1_scores == max(f1_scores))[1],2]
  seuil_opti[[k]] <- tune_glmnet[which(f1_scores == max(f1_scores))[1],3]
  
  print(paste("GLM net fold :", k, "/ 10 - DONE"))
}

alpha_opti
lambda_opti
seuil_opti

alpha_glmnet <- sort(table(alpha_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()
lambda_glmnet <- sort(table(lambda_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()
s_glmnet <- sort(table(seuil_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()

mod_glmnet_opti <- glmnet(x = as.matrix(dta_train[-8]), y = dta_train$Depression_severity, 
                          family = 'binomial',
                          alpha = alpha_glmnet, lambda = lambda_glmnet)

pred_glmnet <- predict(mod_glmnet_opti, newx = as.matrix(dta_test[-8]), type = 'response')
pred_classe_glmnet <- ifelse(pred_glmnet > s_glmnet, 1, 0)

conf_glmnet <- table(dta_test$Depression_severity, pred_classe_glmnet)
conf_glmnet


# SVM
library(e1071)

tune_svm <- expand.grid(C = seq(0.1, 2, length = 5),
                        s = seq(0.05, 0.5, by = 0.05))

C_opti <- rep(0, 10)
seuil_opti <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]
  
  f1_scores <- rep(0, nrow(tune_svm))
  
  for (i in 1:nrow(tune_svm)){
    mod_svm <- svm(Depression_severity ~., data = train, cost = tune_svm[i,1], probability = T)
    
    pred <- predict(mod_svm, newdata = valid, probability = T) |>
      attributes()
    pred_classe <- apply(pred$probabilities, MARGIN = 1, FUN = function(x, s) return(ifelse(x[1] > s, 1, 0)),
                         s = tune_svm[i,2])
    
    f1_scores[i] <- f1(as.numeric(valid$Depression_severity) - 1, 
                       as.numeric(pred_classe))
  }
  
  C_opti[[k]] <- tune_svm[which(f1_scores == max(f1_scores))[1],1]
  seuil_opti[[k]] <- tune_svm[which(f1_scores == max(f1_scores))[1],2]
  
  print(paste("SVM fold :", k, "/ 10 - DONE"))
}

C_opti
seuil_opti

C_svm <- sort(table(C_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()
s_svm <- sort(table(seuil_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()


mod_svm_opti <- svm(Depression_severity ~., data = dta_train, cost = C_svm, probability = T)
pred_svm <- predict(mod_svm_opti, newdata = dta_test, probability = T) |>
  attributes()
pred_classe_svm <- apply(pred_svm$probabilities, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                     s = s_svm)

conf_svm <- table(dta_test$Depression_severity, pred_classe_svm)
conf_svm


# Random forest
library(ranger)

tune_rf <- expand.grid(mtry = 2:10,
                      min.node.size = 2:10,
                      s = seq(0.05, 0.5, by = 0.05))

segs <- pls::cvsegments(nrow(dta_train), 10)
  
for (i in 1:nrow(tune_rf)){
  mod_rf <- ranger(Depression_severity ~., data = dta_train, mtry = tune_rf[i,1], 
                   min.node.size = tune_rf[i,2], probability = T)
  
  pred <- mod_rf$predictions
  pred_classe <- apply(pred, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                       s = tune_rf[i,3])
  
  f1_scores[i] <- f1(as.numeric(dta_train$Depression_severity) - 1, 
                     as.numeric(pred_classe))
  
  print(paste("Random forest :", i, "/ 810 - DONE"))
}
  
mtry_rf <- tune_rf[which(f1_scores == max(f1_scores))[1],1]
node_size_rf <- tune_rf[which(f1_scores == max(f1_scores))[1],2]
s_rf <- tune_rf[which(f1_scores == max(f1_scores))[1],3]

mod_rf_opti <- ranger(Depression_severity ~., data = dta_train, mtry = mtry_rf,
                       min.node.size = node_size_rf, probability = T)
pred_rf <- predict(mod_rf_opti, data = dta_test, type = 'response')
pred_classe_rf <- apply(pred_rf$predictions, MARGIN = 1, FUN = function(x) return(ifelse(x[2] > s_rf, 1, 0)))

conf_rf <- table(dta_test$Depression_severity, pred_classe_rf)
conf_rf


########################################
# Comparaison des performances
########################################

# ici : F1-score, taux de faux positifs, taux de faux négatifs

tab_perf <- data.frame(Méthode = c("kNN", "GLM", "GLM net", "SVM", "Random forest"),
              F1_score = c(f1_tab(conf_knn),
                           f1_tab(conf_glm),
                           f1_tab(conf_glmnet),
                           f1_tab(conf_svm),
                           f1_tab(conf_rf)),
              Faux_positifs = c(tfp(conf_knn),
                                tfp(conf_glm),
                                tfp(conf_glmnet),
                                tfp(conf_svm),
                                tfp(conf_rf)),
              Faux_negatifs = c(tfn(conf_knn),
                                tfn(conf_glm),
                                tfn(conf_glmnet),
                                tfn(conf_svm),
                                tfn(conf_rf)))

tab_perf

save(tab_perf, file = "tab_perf.rData")


########################################
# Sélection de variables
########################################

library(RcmdrMisc)

# Stepwise avec critères AIC et BIC
mod_step <- stepwise(mod_glm_opti, direction = 'backward', criterion = 'AIC')
mod_step_bic <- stepwise(mod_glm_opti, direction = 'backward', criterion = 'BIC')

mod_step$formula
mod_step_bic$formula

# Optimisation du seuil pour chaque nouveau modèle, par validation croisée 10-fold
tune_glm <- expand.grid(s = seq(0.05, 0.5, by = 0.05))

seuil_opti_aic <- rep(0, 10)
seuil_opti_bic <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]
  
  f1_scores_aic <- rep(0, nrow(tune_glm))
  f1_scores_bic <- rep(0, nrow(tune_glm))
  
  for (i in 1:nrow(tune_glm)){
    mod_step <- glm(mod_step$formula, data = train, family = 'binomial')
    mod_step_bic <- glm(mod_step_bic$formula, data = train, family = 'binomial')
    
    pred_aic <- predict(mod_step, newdata = valid, type = 'response')
    pred_bic <- predict(mod_step_bic, newdata = valid, type = 'response')
    
    pred_classe_aic <- ifelse(pred_aic >= tune_glm[i,1], 1, 0)
    pred_classe_bic <- ifelse(pred_bic >= tune_glm[i,1], 1, 0)
    
    f1_scores_aic[i] <- f1(as.numeric(valid$Depression_severity) - 1,
                           as.numeric(pred_classe_aic))
    f1_scores_bic[i] <- f1(as.numeric(valid$Depression_severity) - 1,
                           as.numeric(pred_classe_bic))
  }
  
  seuil_opti_aic[[k]] <- tune_glm[which(f1_scores_aic == max(f1_scores_aic))[1],1]
  seuil_opti_bic[[k]] <- tune_glm[which(f1_scores_bic == max(f1_scores_bic))[1],1]
  print(paste("GLM fold :", k, "/ 10 - DONE"))
}

seuil_opti_aic
seuil_opti_bic

# Prédiction avec le seuil optimal
s_aic <- sort(table(seuil_opti_aic), decreasing = T)[1] |> 
  names() |>
  as.numeric()
s_bic <- sort(table(seuil_opti_bic), decreasing = T)[1] |> 
  names() |>
  as.numeric()

pred_aic <- predict(mod_step, newdata = dta_test, type = 'response')
pred_classe_aic <- ifelse(pred_aic >= s_aic, 1, 0)
pred_bic <- predict(mod_step_bic, newdata = dta_test, type = 'response')
pred_classe_bic <- ifelse(pred_bic >= s_bic, 1, 0)

conf_aic <- table(dta_test$Depression_severity, pred_classe_aic)
conf_bic <- table(dta_test$Depression_severity, pred_classe_bic)

conf_glm
conf_aic
conf_bic

# Comparaison avec le modèle complet
tab_perf_glm <- data.frame(Modèle = c("Complet", "Step AIC", "Step BIC"),
                       F1_score = c(f1_tab(conf_glm),
                                    f1_tab(conf_aic),
                                    f1_tab(conf_bic)),
                       Faux_positifs = c(tfp(conf_glm),
                                         tfp(conf_aic),
                                         tfp(conf_bic)),
                       Faux_negatifs = c(tfn(conf_glm),
                                         tfn(conf_aic),
                                         tfn(conf_bic)))

tab_perf_glm
