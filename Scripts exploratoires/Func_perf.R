
f1 <- function(ytrue, ypred){
  tab <- as.matrix(table(ytrue, ypred))
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return(tab[2,2]/ (tab[2,2] + tab[1,2]/2 + tab[2,1]/2) )
  } else if (colnames(tab) == '1'){
    return(tab[2,1] / (tab[2,1] + tab[1,1]/2))
  } else {
    return(0)
  }
}

f1_tab <- function(tab){
  tab <- as.matrix(tab)
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return(tab[2,2]/ (tab[2,2] + tab[1,2]/2 + tab[2,1]/2) )
  } else if (colnames(tab) == '1'){
    return(tab[2,1] / (tab[2,1] + tab[1,1]/2))
  } else {
    return(0)
  }
}

bal_acc <- function(tab){
  tab <- as.matrix(tab)
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return( tab[1,1]/2 + tab[2,2]/2)
  } else if (colnames(tab) == '1'){
    return(tab[2,1]/2)
  } else {
    return(tab[1,1]/2)
  }
}

tfp <- function(tab){
  tab <- as.matrix(tab)
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return( tab[1,2]/sum(tab[,2]) )
  } else if (colnames(tab) == '1'){
    return(tab[1,1]/sum(tab))
  } else {
    return(0)
  }
}

tfn <- function(tab){
  tab <- as.matrix(tab)
  if (ncol(tab) == 2 & nrow(tab) == 2) {
    return( tab[2,1]/ sum(tab[2,]) )
  } else if (colnames(tab) == '1'){
    return(0)
  } else {
    return(tab[2,1]/sum(tab))
  }
}
