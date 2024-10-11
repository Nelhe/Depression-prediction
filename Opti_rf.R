
library(ranger)
library(caret)

# Chargement des données ----------------

load("dta.RData")
dta <- dta[, -c(1, 10:14)]      # retrait des colonnes inutiles

set.seed(123)
ind_train <- sample(1:nrow(dta), 0.6*nrow(dta))   # 60/40

dta_train <- dta[ind_train,]
dta_test <- dta[-ind_train,]

summary(dta_train$Depression_level)     # vérification que la répartition est similaire entre train et test
summary(dta_test$Depression_level)


# Optimisation de la forêt ----------------

param <- expand.grid(mtry = 2:10,
                     min.node.size = 2:10,
                     splitrule = 'variance')
ctrl <- trainControl(method = 'oob')   # erreur OOB pour une forêt 

mod_rf <- train(Depression_level ~., data = dta_train,
                method = 'ranger', trControl = ctrl, tuneGrid = param)
mod_rf$results
mod_rf$bestTune

pred_rf <- predict(mod_rf, newdata = dta_test)
sqrt(mean((pred_rf - dta_test$Depression_level)^2))     # calcul de la RMSEP
summary(pred_rf - dta_test$Depression_level)      # distribution des erreurs

dta_err <- dta_test
dta_err$err <- pred_rf - dta_test$Depression_level
dta_err_abs <- dta_err
dta_err_abs$err <- abs(pred_rf - dta_test$Depression_level)
FactoMineR::condes(dta_err_abs, 23)     # lien entre erreurs et autres variables


# Transformation de variable --------------

dta2 <- dta
dta2$Depression_level <- sqrt(dta2$Depression_level)    # on passe à la racine carrée
boxplot(dta$Depression_level)
boxplot(dta2$Depression_level)      # répartition plus uniforme

dta2_train <- dta2[ind_train,]
dta2_test <- dta2[-ind_train,]

ctrl2 <- trainControl(method = 'oob')

mod_rf2 <- train(Depression_level ~., data = dta2_train,
                method = 'ranger', trControl = ctrl2, tuneGrid = param)
mod_rf2$results
mod_rf2$bestTune

pred_rf2 <- predict(mod_rf2, newdata = dta2_test)
sqrt(mean((pred_rf2 - dta2_test$Depression_level)^2))     # RMSEP sur la racine carrée
summary(pred_rf2 - dta2_test$Depression_level)
sqrt(mean((pred_rf2^2 - dta_test$Depression_level)^2))    # RMSEP sur la vraie valeur

dta2_err <- dta2_test
dta2_err$err <- abs(pred_rf2 - dta2_test$Depression_level)
FactoMineR::condes(dta2_err, 23)

dta2_err2 <- dta2_test
dta2_err2$err <- abs(pred_rf2^2 - dta_test$Depression_level)
FactoMineR::condes(dta2_err2, 23)


# Visualisation de l'erreur ----------------

library(ggplot2)

ggplot(dta_err) + aes(x = as.factor(Depression_level), y = err) +
  geom_boxplot() +
  geom_hline(yintercept = 0, color = 'red', linetype = 'dashed')

dta_err$fitted <- pred_rf

ggplot(dta_err) + aes(x = Depression_level, y = fitted) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = 'dashed')


# Pseudo-SMOTE sur des niveaux de dépression égaux à 0 --------------

table(as.factor(dta_train$Depression_level))
# idée : passer de 429 valeurs "0" à 150 (valeur choisie arbitrairement)

dta_train_no0 <- dta_train[dta_train$Depression_level != 0,]
dta_train_0 <- dta_train[dta_train$Depression_level == 0,]

library(magrittr)

dta_train_reech <- dta_train_0[sample(1:nrow(dta_train_0), 150), ] |>
  dplyr::bind_rows(dta_train_no0) %>%
  .[sample(1:nrow(.), nrow(.)),]

mod_rf_reech <- train(Depression_level ~., data = dta_train_reech,
                method = 'ranger', trControl = ctrl, tuneGrid = param)
mod_rf_reech$results
mod_rf_reech$bestTune

pred_rf_reech <- predict(mod_rf_reech, newdata = dta_test)
sqrt(mean((pred_rf_reech - dta_test$Depression_level)^2))     # calcul de la RMSEP
summary(pred_rf_reech - dta_test$Depression_level)      # distribution des erreurs
boxplot(pred_rf_reech - dta_test$Depression_level)

dta_err <- data.frame(Valeur = c(dta_test$Depression_level, pred_rf_reech),
                      Type = rep(c("Observé", "Prédit"), each = 1074))
ggplot(dta_err) + aes(y = Valeur, x = Type) + 
  geom_boxplot()
