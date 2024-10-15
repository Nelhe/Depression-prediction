
################################
# Visualisation des données
################################

library(ggplot2)

load("dta.RData")
dta <- dta[, -c(1, 10:14)]    # suppression des variables inutiles

summary(dta)

## Visualisation de la variable à prédire
ggplot(dta) + aes(y = Depression_level) +
  geom_boxplot(alpha = 0.8) +
  labs(title = "Depression level", y = NULL)

## AFDM
dta_clair <- dta

# Réajustement des niveaux pour la clarté
levels(dta_clair$Gender) <- c('Male', 'Female')
levels(dta_clair$Job) <- c('Nurse', 'Physician')

for (i in c(4:7,9:22)){
  # niveaux oui/non :
  if (length(levels(dta_clair[,i])) == 2) {
    dta_clair[,i] <- forcats::fct_recode(dta_clair[,i], No = '1', Yes = '2')
    levels(dta_clair[,i]) <- paste(names(dta_clair)[i], levels(dta_clair[,i]), sep = '_')
  } else {
  # niveaux pour statut familial :
    dta_clair[,i] <- forcats::fct_recode(dta_clair[,i], Single = '1', Married = '2', Divorced = '3', Widowhood = '4')
  }
}

summary(dta_clair)  # vérification des niveaux

res_famd <- FactoMineR::FAMD(dta_clair, sup.var = 8, graph = F)

# graphique des variables quali :
factoextra::fviz_famd_var(res_famd, "quali.var", col.var = 'contrib', repel = T)
# graphique des variables :
factoextra::fviz_famd_var(res_famd, "var", col.var = 'contrib')


################################
# Entraînement des algorithmes
################################

set.seed(123)

ind_train <- sample(1:nrow(dta), 0.8*nrow(dta))   # 80/20

dta_train <- dta[ind_train,]
dta_test <- dta[-ind_train,]

library(caret)

ctrl <- trainControl(method = "LGOCV", number = 10)

# k plus proches voisins
tune_knn <- expand.grid(k = 1:100)

mod_knn <- train(Depression_level ~ ., data = dta_train,
                 method = "knn", trControl = ctrl, tuneGrid = tune_knn)

# GLM
mod_glm <- train(Depression_level ~ ., data = dta_train,
                 method = "glm", family = gaussian, trControl = ctrl)

# GLM net
tune_glmnet <- expand.grid(alpha = seq(0, 1, length = 20),  
                           lambda = seq(0.01, 1.1, length = 20))
mod_glmnet <- train(Depression_level ~ ., data = dta_train, 
                     method = "glmnet", trControl = ctrl, 
                    tuneGrid = tune_glmnet) 

# SVM
# tune_svm <- expand.grid(C = seq(0, 2, length = 5))
# prend trop longtemps à tourner

mod_svm <- train(Depression_level ~ ., data = dta_train, 
                 method = "svmLinear", trControl = ctrl) 
                # tuneGrid = tune_svm

# Random forest
tune_rf <- expand.grid(mtry = 2:10,
                     min.node.size = 2:10,
                     splitrule = 'variance')
ctrl_rf <- trainControl(method = 'oob')   # erreur OOB pour une forêt 

mod_rf <- train(Depression_level ~., data = dta_train,
                method = 'ranger', trControl = ctrl_rf, tuneGrid = tune_rf)


########################################
# Comparaison des erreurs de prédiction
########################################

# Calcul des prédictions du meilleur modèle pour chaque méthode
pred_knn <- predict(mod_knn, newdata = dta_test)
pred_glm <- predict(mod_glm, newdata = dta_test)
pred_glmnet <- predict(mod_glmnet, newdata = dta_test)
pred_svm <- predict(mod_svm, newdata = dta_test)
pred_rf <- predict(mod_rf, newdata = dta_test)

# Calcul des RMSEP
tab_rmsep <- data.frame(Method = c('knn', 'GLM', 'GLM net', 'SVM', 'Random forest'),
                        RMSEP = c(sqrt(mean((pred_knn - dta_test$Depression_level)^2)),
                                  sqrt(mean((pred_glm - dta_test$Depression_level)^2)),
                                  sqrt(mean((pred_glmnet - dta_test$Depression_level)^2)),
                                  sqrt(mean((pred_svm - dta_test$Depression_level)^2)),
                                  sqrt(mean((pred_rf - dta_test$Depression_level)^2))))
tab_rmsep

# Répartition des erreurs
dta_err <- data.frame(Observed = dta$Depression_level, 
                      KNN = pred_knn,
                      GLM = pred_glm,
                      GLMnet = pred_glmnet,
                      SVM = pred_svm,
                      RF = pred_rf) |>
  tidyr::pivot_longer(Observed:RF, names_to = 'Method', values_to = 'Values')
dta_err$Method <- as.factor(dta_err$Method) |>
  forcats::fct_relevel('Observed')

ggplot(dta_err) + aes(x = Method, y = Values) +
  geom_boxplot()

# Graphique valeurs observées / valeurs prédites
ggplot(data.frame(Observed = dta_test$Depression_level,
                  Predicted = pred_svm)) + 
  aes(x = Observed, y = Predicted, color = Observed) +
  geom_point() +
  geom_abline(a = 0, b = 1, linetype = 'dashed') +
  labs(title = 'Predicted vs Observed')

# Proportion de valeurs sur- et sous-estimées
mean(dta_test$Depression_level > pred_svm)
mean(dta_test$Depression_level < pred_svm)

# Test de la relation linéaire
test_lien <- lm(Predicted ~ Observed, data = data.frame(Observed = dta_test$Depression_level,
                                                         Predicted = pred_svm))
summary(test_lien) # lien linéaire pas à écarter, on teste les seuils


########################################
# Choix d'un seuil de décision
########################################

# Méthodes évaluées : GLM, SVM

library(pROC)

# Score > 4.5 : dépression légère (d'après le DASS-21)
depr_obs <- ifelse(dta_test$Depression_level*2 > 9, 1, 0)

# Test de seuils pour les valeurs prédites
seuils <- seq(1, 10, by = 0.5)

acc_glm <- rep(0, length(seuils))
f1_glm <- rep(0, length(seuils))
aucs_glm <- rep(0, length(seuils))

acc_svm <- rep(0, length(seuils))
f1_svm <- rep(0, length(seuils))
aucs_svm <- rep(0, length(seuils))

for (i in 1:length(seuils)){
  predic_svm <- ifelse(pred_svm > seuils[i], 1, 0)
  predic_glm <- ifelse(pred_glm > seuils[i], 1, 0)
  
  acc_glm[i] <- mean(depr_obs == predic_glm)
  acc_svm[i] <- mean(depr_obs == predic_svm)
  
  mat_conf_glm <- as.matrix(table(depr_obs, predic_glm))
  mat_conf_svm <- as.matrix(table(depr_obs, predic_svm))
  f1_glm[i] <- 2*mat_conf_glm[2,2]/(2*mat_conf_glm[2,2] + mat_conf_glm[1,2] + mat_conf_glm[2,1])
  f1_svm[i] <- 2*mat_conf_svm[2,2]/(2*mat_conf_svm[2,2] + mat_conf_svm[1,2] + mat_conf_svm[2,1])
  
  roc_glm <- roc(depr_obs, predic_glm)
  roc_svm <- roc(depr_obs, predic_svm)
  aucs_glm[i] <- auc(roc_glm)
  aucs_svm[i] <- auc(roc_svm)
}

# Représentation de l'évolution des critères pour chaque méthode
tab_crit <- data.frame(Threshold = c(seuils, seuils),
                       Accuracy = c(acc_glm, acc_svm),
                       F1_score = c(f1_glm, f1_svm),
                       AUC = c(aucs_glm, aucs_svm),
                       Method = rep(c('GLM', 'SVM'), each = length(seuils)))

ggplot(tab_crit) + aes(x = Threshold, y = Accuracy, color = Method) +
  geom_point() +
  geom_line() +
  labs(title = "Accuracy")

ggplot(tab_crit) + aes(x = Threshold, y = F1_score, color = Method) +
  geom_point() +
  geom_line() +
  labs(title = "F1 score")

ggplot(tab_crit) + aes(x = Threshold, y = AUC, color = Method) +
  geom_point() +
  geom_line() +
  labs(title = "AUC")

# Détection des seuils optimisés pour chaque critère
seuils[which(acc_glm == max(acc_glm))]
seuils[which(f1_glm == max(f1_glm))]
seuils[which(aucs_glm == max(aucs_glm))]

seuils[which(acc_svm == max(acc_svm))]
seuils[which(f1_svm == max(f1_svm))]
seuils[which(aucs_svm == max(aucs_svm))]

# Matrices de confusion pour les seuils optimaux choisis
table(depr_obs, ifelse(pred_glm > 4, 1, 0))
table(depr_obs, ifelse(pred_svm > 3, 1, 0))


########################################
# Sélection de variables pour le GLM
########################################

mod_complet <- lm(Depression_level ~ ., data = dta_train)

mod_step <- RcmdrMisc::stepwise(mod_complet, direction = 'backward')
mod_step_bic <- RcmdrMisc::stepwise(mod_complet, direction = 'backward', criterion = 'BIC')

pred_glm_reduit <- predict(mod_step, newdata = dta_test) 
sqrt(mean((pred_glm_reduit - dta_test$Depression_level)^2))

dta_err_glm <- data.frame(Values = c(pred_glm, pred_glm_reduit),
                          Method = rep(c('GLM', 'GLM step'), each = length(pred_glm)))

ggplot(dta_err_glm) + aes(x = Method, y = Values) +
  geom_boxplot() +
  labs(title = "GLM complet vs GLM réduit")

table(depr_obs, ifelse(pred_glm_reduit > 4, 1, 0))
