summary(dta)
names(dta)
library(DataExplorer)
library(ggplot2)
?DataExplorer
create_report(dta)

max(dta$Depression_level)
min(dta$Depression_level)

ggplot(data=dta, aes(x=Country, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=Gender, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=Age, y=Depression_level))+
  geom_smooth()

ggplot(data=dta, aes(x=Manager_position, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=Job, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=more_work, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=additional_workload , y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=overtime, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=Job, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=work_stress , y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=work_conflicts, y=Depression_level))+
  geom_boxplot()

ggplot(data=dta, aes(x=afraid_family, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=people_avoid_me, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=afraid_others, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=working_attitude, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=insufficient_employees, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=appreciation_employer, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=appreciation_society, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=appreciation_govt , y=Depression_level))+
  geom_boxplot()

ggplot(data=dta, aes(x=COVID_frontline, y=Depression_level))+
  geom_boxplot()

ggplot(data=dta, aes(x=Family_status, y=Depression_level))+
  geom_boxplot()
ggplot(data=dta, aes(x=Children, y=Depression_level))+
  geom_boxplot()

#Country Gender Age JobManager_position COVID_frontline Family_status Children  Depression
# Trouver les variables les plus significatives pour déterminer la dépression
dta2 <- dta[,-c(10:14)]
dta2 <- dta2[,-c(1)]
names(dta2)

# 0,75 pour p contrôle 
# méthode LGOCV pour k fold

library(caret)
# Diviser les données en ensembles d'entraînement (80%) et de test (20%)
set.seed(123)
trainIndex <- createDataPartition(dta2$Depression_level, p = 0.8, list = FALSE)
data_train <- dta2[trainIndex, ]
data_test <- dta2[-trainIndex, ]

# Définir le contrôle d'entraînement
fitControl <- trainControl(
  method = "LGOCV",  # Validation croisée répétée
  number = 10,            # 10-fold CV
  p = 0.7
)

# Entraîner le modèle KNN
mod_knn_sans_TuneGrid <- train(
  Depression_level ~ .,  # Utiliser toutes les variables comme prédicteurs
  data = data_train,
  method = "knn",
  trControl = fitControl
)
print(mod_knn_sans_TuneGrid)
plot(mod_knn_sans_TuneGrid)

# Définir la grille de paramètres pour k
tuneGrid <- data.frame(k = 1:100)
tuneGrid 
# tuneGrid définit les valeurs de l'hyperparamètre que vous voulez tester pour votre
# modèle KNN. Dans ce cas, il s'agit du paramètre k, qui représente le nombre de voisins
# à considérer.

# Recherche de la meilleure valeur :
# Caret va entraîner un modèle KNN pour chaque valeur de k spécifiée dans tuneGrid, 
# en utilisant la méthode de validation croisée définie dans trainControl.
# Optimisation automatique :
# Pour chaque valeur de k, caret évaluera la performance du modèle (généralement en
# termes d'erreur ou de précision). Il sélectionnera ensuite automatiquement la valeur 
# de k qui donne les meilleures performances.
# Exploration systématique :
# En définissant une grille de 1 à 20, vous vous assurez d'explorer un large éventail
# de possibilités pour le paramètre k, de très peu de voisins (k=1) à beaucoup (k=20).

# Entraîner le modèle KNN
mod_knn <- train(
  Depression_level ~ .,  # Utiliser toutes les variables comme prédicteurs
  data = data_train,
  method = "knn",
  trControl = fitControl,
  tuneGrid = tuneGrid
)

# Afficher les résultats
print(mod_knn)
plot(mod_knn)
mod_knn$results
# mean absolute error (MAE), smaller indicates a better fit, and a perfect fit is 
# equal to 0.
# le RMSE permet de choisir le meilleur hyperparamètre
# donc ici on choisit k = 17, pour 17 plus proche voisin, car on a la RMSE la 
# plus faible

# Entraîner le modèle KNN
mod_knn <- train(
  Depression_level ~ .,  # Utiliser toutes les variables comme prédicteurs
  data = data_train,
  method = "knn",
  trControl = fitControl,
  tuneGrid = data.frame(k = 17)
)
# Prédire sur l'ensemble de test
predictions <- predict(mod_knn, newdata = data_test)

# la rmse pour les valeurs continues c'est très interprétable 
# plutôt que d'utiliser une autre métrique ( en quali l'accuracy est un bon point
# de départ, quand a des données déséquilibrées, le F1-Score sera bcp plus efficace,
# le F1 sera surtout utilisé pour les var quali binaire, sinon avec plus de 2 modalités,
# on a peut etre le Kappa de Cohen)
predictions

# Évaluer les performances
mse <- mean((data_test$Depression_level - predictions)^2)
# Cette ligne calcule la moyenne des carrés des différences entre les 
# valeurs réelles et prédites. C'est une mesure standard de l'erreur de prédiction.
rmse <- sqrt(mse)
#Elle est souvent préférée car elle est dans la même unité que la variable prédite.
cat("MSE:", mse, "\n") # erreur moyenne
cat("RMSE:", rmse, "\n")
mae <- mean(abs(data_test$Depression_level - predictions))
cat("MAE:", mae, "\n")
# La MAE (Mean Absolute Error) est une autre métrique utile qui est moins sensible 
# aux valeurs extrêmes que la RMSE.

# Visualiser les prédictions vs. les valeurs réelles
plot(data_test$Depression_level, predictions, 
     main = "Prédictions vs. Valeurs réelles",
     xlab = "Valeurs réelles", ylab = "Prédictions")
# Dans un modèle parfait, tous les points seraient alignés sur une ligne 
# droite diagonale (y = x).
# On observe une tendance positive générale, ce qui est bon signe. 
# Cela signifie que le modèle capture une relation entre les valeurs réelles et prédites.
# Cependant, la dispersion est assez large, indiquant que les prédictions ne sont 
# pas très précises.
# Les prédictions semblent se concentrer entre 2 et 8 environ, alors que les valeurs
# réelles vont de 0 à 20. Cela suggère que le modèle a tendance à "moyenner" ses
# prédictions, évitant les valeurs extrêmes.
# On a Surestimation des valeurs basses et Sous-estimation des valeurs élevées
abline(0, 1, col = "red")
# Si la majorité des points se trouve au-dessus de la ligne, cela indique une 
# tendance générale à la surestimation.

dta_err <- data.frame(Valeur = c(data_test$Depression_level, predictions),
                      Type = rep(c("Observé", "Prédit"), each = 536))
ggplot(dta_err) + aes(y = Valeur, x = Type) + 
  geom_boxplot()
