
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
factoextra::fviz_famd_var(res_famd, "var", col.var = 'contrib', repel = T)


################################
# Entraînement des algorithmes
################################

set.seed(123)

ind_train <- sample(1:nrow(dta), 0.8*nrow(dta))   # 80/20

dta_train <- dta[ind_train,]
dta_test <- dta[-ind_train,]

library(caret)

ctrl <- trainControl(method = "cv", number = 10)

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

ggplot(dta_err) + aes(x = Method, y = Values, fill = Method) +
  geom_boxplot() +
  scale_fill_brewer(palette = "Set2", type = 'qual') +
  labs(title = "Observed values VS predicted values")

# Graphique valeurs observées / valeurs prédites
ggplot(data.frame(Observed = dta_test$Depression_level,
                  Predicted = pred_glm)) + 
  aes(x = Observed, y = Predicted, color = Observed) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = 'dashed') +
  labs(title = 'Predicted vs Observed')

# Proportion de valeurs sur- et sous-estimées
mean(dta_test$Depression_level > pred_glm)
mean(dta_test$Depression_level < pred_glm)

# Test de la relation linéaire
test_lien <- lm(Predicted ~ Observed, data = data.frame(Observed = dta_test$Depression_level,
                                                         Predicted = pred_glm))
summary(test_lien) # lien linéaire pas à écarter, on teste les seuils


########################################
# Choix d'un seuil de décision
########################################

# Méthode évaluée : GLM 

library(pROC)

# Score > 4.5 : dépression légère (d'après le DASS-21)
depr_obs <- ifelse(dta_test$Depression_level*2 > 9, 1, 0)

# Test de seuils pour les valeurs prédites
seuils <- seq(1, 10, by = 0.5)

acc_glm <- rep(0, length(seuils))
f1_glm <- rep(0, length(seuils))
aucs_glm <- rep(0, length(seuils))

for (i in 1:length(seuils)){
  predic_glm <- ifelse(pred_glm > seuils[i], 1, 0)
  
  acc_glm[i] <- mean(depr_obs == predic_glm)
  
  mat_conf_glm <- as.matrix(table(depr_obs, predic_glm))
  f1_glm[i] <- 2*mat_conf_glm[2,2]/(2*mat_conf_glm[2,2] + mat_conf_glm[1,2] + mat_conf_glm[2,1])
  
  roc_glm <- roc(depr_obs, predic_glm)
  aucs_glm[i] <- auc(roc_glm)
}

# Détection des seuils optimisés
seuils[which(acc_glm == max(acc_glm))]
seuils[which(f1_glm == max(f1_glm))]
seuils[which(aucs_glm == max(aucs_glm))]

# Matrice de confusion pour le seuil optimal choisi
library(yardstick)

table(depr_obs, ifelse(pred_glm > 4, 1, 0)) |>
  conf_mat() |>
  autoplot(type = 'heatmap') + 
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  labs(x = "Valeurs prédites", y = "Valeurs observées")


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

ggplot(dta_err_glm) + aes(x = Method, y = Values, fill = Method) +
  geom_boxplot() +
  scale_fill_brewer(palette = 'Set2') +
  labs(title = "GLM complet vs GLM réduit") 

table(as.factor(depr_obs), ifelse(pred_glm_reduit > 4, 1, 0)) |>
  conf_mat() |>
  autoplot(type = 'heatmap') + 
  scale_fill_gradient(low="#D6EAF8",high = "#2E86C1") +
  labs(x = "Valeurs prédites", y = "Valeurs observées")
