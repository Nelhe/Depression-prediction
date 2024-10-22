
################################
# Visualisation des données
################################

library(ggplot2)

load("dta.RData")
dta <- dta[, -c(1, 9:11, 13:14)] |>
  dplyr::mutate(Depression_severity = as.factor(ifelse(Depression_severity %in% c(0,1), 0, 1)))

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

f1 <- function(ypred, ytrue){
  tab <- as.matrix(table(ytrue, ypred))
  
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return(tab[2,2]/ (tab[2,2] + tab[1,2]/2 + tab[2,1]/2) )
  } else if (colnames(ta) == '0'){
    return(0)
  } else {
    return(tab[2,1] / (tab[2,1] + tab[1,1]/2))
  }
}


# k plus proches voisins
tune_knn <- expand.grid(k = 1:50,
                        s = seq(0.2, 0.5, by = 0.05))

segs <- pls::cvsegments(nrow(dta_train), 10)

k_opti <- rep(0, 10)
seuil_opti <- rep(0, 10)

for (k in 1:10){
  train <- dta_train[-segs[[k]],]
  valid <- dta_train[segs[[k]],]
  
  f1_scores <- rep(0, nrow(tune_knn))
  
  for (i in 1:nrow(tune_knn)){
    mod_knn <- kknn::kknn(Depression_severity ~., train = train, test = valid, k = tune_knn[i,1])
    
    pred_classe <- apply(mod_knn$prob, MARGIN = 1, FUN = function(x, s) return(ifelse(x[2] > s, 1, 0)),
                         s = tune_knn[i,2])
    
    f1_scores[i] <- f1(as.numeric(valid$Depression_severity) - 1, 
                       as.numeric(pred_classe))
  }
  
  k_opti[[k]] <- tune_knn[which(f1_scores == max(f1_scores))[1],1]
  seuil_opti[[k]] <- tune_knn[which(f1_scores == max(f1_scores))[1],2]
  
  print(k)
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

table(dta_test$Depression_severity, pred_classe_knn)


# GLM
tune_glm <- expand.grid(s = seq(0.05, 0.5, by = 0.05))

segs <- pls::cvsegments(nrow(dta_train), 10)

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
  print(k)
}

seuil_opti

s_glm <- sort(table(seuil_opti), decreasing = T)[1] |> 
  names() |>
  as.numeric()

mod_glm_opti <- glm(Depression_severity ~., data = dta_train, family = 'binomial')
pred_glm <- predict(mod_glm_opti, newdata = dta_test, type = 'response')
pred_classe_glm <- ifelse(pred_glm >= s_glm, 1, 0)

table(dta_test$Depression_severity, pred_classe_glm)

# LDA


# SVM
# tune_svm <- expand.grid(C = seq(0.1, 2, length = 5))


# Random forest
library(ranger)

tune_rf <- expand.grid(mtry = 2:10,
                      min.node.size = 2:10,
                      s = seq(0.05, 0.5, by = 0.05))

segs <- pls::cvsegments(nrow(dta_train), 10)

oob_error <- rep(0, nrow(tune_rf))
  
for (i in 1:nrow(tune_rf)){
  mod_rf <- ranger(Depression_severity ~., data = dta_train, mtry = tune_rf[i,1], 
                   min.node.size = tune_rf[i,2], probability = T)
  
  oob_error[i] <- mod_rf$prediction.error
  
  print(i)
}
  
mtry_rf <- tune_rf[which(oob_error == min(oob_error))[1],1]
node_size_rf <- tune_rf[which(oob_error == min(oob_error))[1],2]
s_rf <- tune_rf[which(oob_error == min(oob_error))[1],3]

mod_rf_opti <- ranger(Depression_severity ~., data = dta_train, mtry = mtry_rf,
                       min.node.size = node_size_rf, probability = T)
pred_rf <- predict(mod_rf_opti, data = dta_test, type = 'response')
pred_classe_rf <- apply(pred_rf$predictions, MARGIN = 1, FUN = function(x) return(ifelse(x[2] > s_rf, 1, 0)))

table(dta_test$Depression_severity, pred_classe_rf)

########################################
# Comparaison des erreurs de prédiction
########################################


