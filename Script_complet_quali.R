
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

source("Scripts exploratoires/Func_perf.R")


# k plus proches voisins
library(kknn)

tune_knn <- expand.grid(k = seq(25, 75, by = 5),
                        s = seq(0.2, 0.5, by = 0.05))

f1_score_knn <- rep(0, nrow(tune_knn))

for (i in 1:nrow(tune_knn)){
  preds <- rep(0, nrow(dta_train))
  
  for (k in 1:10){
    train <- dta_train[-segs[[k]],]
    valid <- dta_train[segs[[k]],]
    
    mod_knn <- kknn(Depression_severity ~., train = train, test = valid, k = tune_knn[i,1])
    
    preds[segs[[k]]] <- apply(mod_knn$prob, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                         s = tune_knn[i,2])
  }
  
  f1_score_knn[i] <- f1(as.numeric(dta_train$Depression_severity) - 1,
                 preds)
  
  print(paste(i, nrow(tune_knn), sep = '/'))
}

perf_knn <- data.frame(Méthode = 'kNN',
                       tune_knn,
                       F1_score = f1_score_knn) |>
  dplyr::arrange(desc(F1_score))

perf_knn


# GLM
tune_glm <- expand.grid(s = seq(0.05, 0.5, by = 0.05))

f1_score_glm <- rep(0, nrow(tune_glm))

for (i in 1:nrow(tune_glm)){
  preds <- rep(0, nrow(dta_train))
  
  for (k in 1:10){
    train <- dta_train[-segs[[k]],]
    valid <- dta_train[segs[[k]],]
    
    mod_glm <- glm(Depression_severity ~., data = train, family = 'binomial')
    pred <- predict(mod_glm, newdata = valid, type = 'response')
    preds[segs[[k]]] <- ifelse(pred >= tune_glm[i,1], 1, 0)
  }
  
  f1_score_glm[i] <- f1(as.numeric(dta_train$Depression_severity) - 1,
                    preds)
  
  print(paste(i, nrow(tune_glm), sep = '/'))
}

perf_glm <- data.frame(Méthode = 'GLM',
                       tune_glm,
                       F1_score = f1_score_glm) |>
  dplyr::arrange(desc(F1_score))

perf_glm


# GLM net
library(glmnet)

tune_glmnet <- expand.grid(alp = seq(0, 1, length = 5),
                           lb = seq(1e-4, 1e-1, length = 5),
                           s = seq(0.05, 0.5, by = 0.05))

f1_score_glmnet <- rep(0, nrow(tune_glmnet))

for (i in 1:nrow(tune_glmnet)){
  preds <- rep(0, nrow(dta_train))
  
  for (k in 1:10){
    train <- dta_train[-segs[[k]],]
    valid <- dta_train[segs[[k]],]
    
    mod_glmnet <- glmnet(x = as.matrix(train[-8]), y = train$Depression_severity, 
                         family = 'binomial',
                         alpha = tune_glmnet[i,1], lambda = tune_glmnet[i,2])
    
    pred <- predict(mod_glmnet, newx = as.matrix(valid[-8]), type = 'response')
    preds[segs[[k]]] <- ifelse(pred > tune_glmnet[i,3], 1, 0)
  }
  
  f1_score_glmnet[i] <- f1(as.numeric(dta_train$Depression_severity) - 1,
                        preds)
  
  print(paste(i, nrow(tune_glmnet), sep = '/'))
}

perf_glmnet <- data.frame(Méthode = 'GLMnet',
                       tune_glmnet,
                       F1_score = f1_score_glmnet) |>
  dplyr::arrange(desc(F1_score))

perf_glmnet


# SVM
library(e1071)

tune_svm <- expand.grid(C = seq(0.1, 2, length = 5),
                        s = seq(0.05, 0.5, by = 0.05))

f1_score_svm <- rep(0, nrow(tune_svm))

for (i in 1:nrow(tune_svm)){
  preds <- rep(0, nrow(dta_train))
  
  for (k in 1:10){
    train <- dta_train[-segs[[k]],]
    valid <- dta_train[segs[[k]],]
    
    mod_svm <- svm(Depression_severity ~., data = train, cost = tune_svm[i,1], probability = T)
    
    pred <- predict(mod_svm, newdata = valid, probability = T) |>
      attributes()
    preds[segs[[k]]] <- apply(pred$probabilities, MARGIN = 1, FUN = function(x, s) return(ifelse(x[1] > s, 1, 0)),
                         s = tune_svm[i,2])
  }
  
  f1_score_svm[i] <- f1(as.numeric(dta_train$Depression_severity) - 1,
                           preds)
  
  print(paste(i, nrow(tune_svm), sep = '/'))
}

perf_svm <- data.frame(Méthode = 'SVM',
                          tune_svm,
                          F1_score = f1_score_svm) |>
  dplyr::arrange(desc(F1_score))

perf_svm


# Random forest
library(ranger)

tune_rf <- expand.grid(mtry = 2:10,
                      min.node.size = 2:10,
                      s = seq(0.05, 0.5, by = 0.05))

f1_scores_rf <- rep(0, nrow(tune_rf))
  
for (i in 1:nrow(tune_rf)){
  mod_rf <- ranger(Depression_severity ~., data = dta_train, mtry = tune_rf[i,1], 
                   min.node.size = tune_rf[i,2], probability = T)
  
  pred <- mod_rf$predictions
  pred_classe <- apply(pred, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                       s = tune_rf[i,3])
  
  f1_scores_rf[i] <- f1(as.numeric(dta_train$Depression_severity) - 1, 
                     as.numeric(pred_classe))
  
  print(paste("Random forest :", i, "/ 810 - DONE"))
}
  
perf_rf <- data.frame(Méthode = 'Random forest',
                       tune_rf,
                       F1_score = f1_scores_rf) |>
  dplyr::arrange(desc(F1_score))

perf_rf


########################################
# Comparaison des performances
########################################

# ici : F1-score, taux de faux positifs, taux de faux négatifs

comp_perf <- dplyr::bind_rows(perf_glm, perf_glmnet, perf_knn, perf_rf, perf_svm) |>
  dplyr::arrange(desc(F1_score))

head(comp_perf)

best_mod_overall <- glmnet(x = dta_train[-8], y = dta_train$Depression_severity,
                           family = "binomial", alpha = 0.25, lambda = 0.05005)
best_mod_overall$beta
pred_best <- predict(best_mod_overall, newx = as.matrix(dta_test[-8]), type = 'response')
preds_best_classe <- ifelse(pred_best > 0.35, 1, 0)

conf_fin <- table(dta_test$Depression_severity, preds_best_classe)

save(comp_perf, file = 'comp_perf.RData')

best_mod_2 <- glmnet(x = dta_train[-8], y = dta_train$Depression_severity,
                     family = "binomial", alpha = 0.5, lambda = 0.025075)
best_mod_2$beta

best_mod_3 <- glmnet(x = dta_train[-8], y = dta_train$Depression_severity,
                     family = "binomial", alpha = 0.75, lambda = 0.025075)
best_mod_3$beta

best_mod_4 <- glmnet(x = dta_train[-8], y = dta_train$Depression_severity,
                     family = "binomial", alpha = 0.25, lambda = 0.025075)
best_mod_4$beta

best_mod_5 <- glmnet(x = dta_train[-8], y = dta_train$Depression_severity,
                     family = "binomial", alpha = 1, lambda = 0.025075)
best_mod_5$beta

# Transformer la matrice de confusion en data frame pour ggplot2
conf_fin_df <- as.data.frame(conf_fin)

# Création d'un heatmap avec ggplot2
ggplot(data = conf_fin_df, aes(x = preds_best_classe, y = Var1)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "#f7c6c7", high = "#d73027") +
  geom_text(aes(label = Freq), vjust = 1) +  # Ajouter les fréquences dans chaque case
  labs(title = "Matrice de confusion - GLM net", 
       x = "Prédictions", 
       y = "Valeurs réelles") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),  # Centrer le titre
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12))
