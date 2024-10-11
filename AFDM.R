
load("dta.RData")
dta_clair <- dta[, -c(1, 10:14)]

for (i in c(1,3:7,9:22)){
  levels(dta_clair[,i]) <- paste(names(dta_clair)[i], levels(dta_clair[,i]), sep = '_')
}

res_famd <- FactoMineR::FAMD(dta_clair, sup.var = 8, graph = F)
factoextra::fviz_famd_var(res_famd_clair, "quali.var", col.var = 'contrib')
factoextra::fviz_famd_var(res_famd_clair, "var", col.var = 'contrib')
