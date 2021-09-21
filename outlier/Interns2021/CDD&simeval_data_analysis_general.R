###############################################################################
### CDD plots from runs                                                     ###
### Simon Carter                                                            ###
### July 2021                                                               ###
###############################################################################

# import libraries
remotes::install_github("UUPharmacometrics/assemblerr")
remotes::install_github("pharmpy/pharmr", ref="main")
pharmr::install_pharmpy()

library(reticulate)
library(tidyverse)
library(jsonlite)
library(tidyjson)
library(here)
library(assemblerr)
library(pharmr)
library(magrittr)
library(xpose)
library(cowplot)

pharmpy <- import("pharmpy")

############ Functions for CDD result extraction ##############################

CDD_extract = function(path,drugs,folders){
  # set up number of folder repeats
  foldersB <- rep(folders,each=length(drugs))
  
  # make a tibble of the all the drug and folder paths 
  drugs_lst <- tibble(paste(path, 
                            drugs, 
                            foldersB, 
                            sep="/"))
  
  # Fill in the read_results paths function
  read_results_wrapper <- function(drug, ...) {
    readres <- pharmpy$results$read_results(drug)
    dofv <- readres$case_results$delta_ofv
    cs <- readres$case_results$cook_score
    jcs <- readres$case_results$jackknife_cook_score
    inf_id <- readres$case_results$dofv_influential
    
    drug_index <- tibble(gsub(pattern = "Y:/lincdd_paper/CDDruns/runs/",
                       x = drug,
                       replacement = ""))
    Drug_name <- str_extract(drug_index, pattern = "(\\w+)")
    CDD_type <- str_remove(drug_index, pattern = "^.*/")
      
    res_out <- tibble(Drug_name, CDD_type, inf_id, dofv, cs, jcs) 
    
    return(res_out)
  }
  
  # map the CDD results for all drugs and folders
  CDD_results <- map(unlist(drugs_lst), read_results_wrapper) %>%
    reduce(full_join,
           by = c('Drug_name', 'CDD_type', 'inf_id', 'dofv', 'cs', 'jcs'))
  
  return(CDD_results)
}

############ Read in and extract all json files from CDD runs ##################
CDD_path <- 'B:/home/USER/alzha120/Training/Datasets_Summer_Interns_2021'

CDD_drugs <- list("Elsherbiny-DA-2010-Lopinavir")

CDD_folders <- list('cdd_nonlinear_run805')

############ Run function with data to extract results + plot  #################

test <- CDD_extract(CDD_path, CDD_drugs, CDD_folders) 

nonlin_cov <- filter(test,CDD_type=='cdd_nonlinear_run805') 
#nonlin_nocov <- filter(test, CDD_type=='cdd_nonlinear_nocov')
#linear_phi <- filter(test, CDD_type=='cdd_linear_phi')

fig1 <- ggplot(data = nonlin_cov) +
  geom_point(aes(x=cs,y=dofv)) +
  geom_smooth(aes(x=cs, y=dofv), method="lm", se = FALSE) +
  facet_wrap(~Drug_name, ncol=8, scales='free_x' ) +
  xlab("Cook Score") +
  theme_bw()
fig1

#fig2 <- ggplot(data = nonlin_nocov) +
  geom_point(aes(x=jcs,y=dofv)) +
  facet_wrap(~Drug_name, ncol=8) +
  xlab("Jackknife Cook Score") +
  theme_bw()

#fig3 <- ggplot(data = linear_phi) +
  geom_point(aes(x=jcs,y=dofv)) +
  facet_wrap(.~Drug_name, ncol=8) +
  xlab("Jackknife Cook Score") +
  theme_bw()

fig1
#fig2
#fig3

############ CDD Predictors analysis  #################

dofv <- test[,'dofv']
predictors_lopinavir <- predictors_lopinavir %>% mutate(dofv)


fig4 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=nobsi_over_avg, y = dofv)) +
  geom_smooth(aes(x=nobsi_over_avg, y = dofv), method="lm", se = FALSE) +
  xlab("obsi_over_avg") +
  theme_bw()
fig4

fig5 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=max_cov_over_sd, y = dofv)) +
  geom_smooth(aes(x=max_cov_over_sd, y = dofv), method="lm", se = FALSE) +
  xlab("max_cov_over_sd") +
  theme_bw()
fig5

fig6 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=CWRES_each_max, y = dofv)) +
  geom_smooth(aes(x=CWRES_each_max, y = dofv), method="lm", se = FALSE) +
  xlab("CWRES_each_max") +
  theme_bw()
fig6

fig7 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=CWRES_each_median, y = dofv)) +
  geom_smooth(aes(x=CWRES_each_median, y = dofv), method="lm", se = FALSE) +
  xlab("CWRES_each_median") +
  theme_bw()
fig7

fig8 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=max_EBE_over_omega_each, y = dofv)) +
  geom_smooth(aes(x=max_EBE_over_omega_each, y = dofv), method="lm", se = FALSE) +
  xlab("max_EBE_over_omega_each") +
  theme_bw()
fig8

fig9 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=OFVRatio, y = dofv)) +
  geom_smooth(aes(x=OFVRatio, y = dofv), method="lm", se = FALSE) +
  xlab("OFVRatio") +
  theme_bw()
fig9

fig10 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=OFVRatio2, y = dofv)) +
  geom_smooth(aes(x=OFVRatio2, y = dofv), method="lm", se = FALSE) +
  xlab("OFVRatio2") +
  theme_bw()
fig10

fig11 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=mean_ETC_omega, y = dofv)) +
  geom_smooth(aes(x=mean_ETC_omega, y = dofv), method="lm", se = FALSE) +
  xlab("mean_ETC_omega") +
  theme_bw()
fig11




plot_grid(fig4, fig5, fig6, fig7, fig8, fig9, fig10 , fig11,
          ncol = 3, nrow = 3)


############ Read in and extract results files from simeval runs ###############
simeval_path <- "B:/home/USER/alzha120/Training/Datasets_Summer_Interns_2021"
simeval_drug <- "/Elsherbiny-DA-2010-Lopinavir"
simeval_file <- "/simeval_nonlinear_run805"

all_path <- paste0(simeval_path, simeval_drug, simeval_file, "/results.csv")

simeval_results <- read.csv(all_path, skip = 34, header = TRUE) %>% drop_na() #change the skip value

############ Simeval Predictors analysis  #################

residuals_values <- simeval_results[,'residual']
predictors_lopinavir <- predictors_lopinavir %>% mutate(residuals_values)



fig12 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=nobsi_over_avg, y = residuals_values)) +
  geom_smooth(aes(x=nobsi_over_avg, y = residuals_values), method="lm", se = FALSE) +
  xlab("obsi_over_avg") +
  theme_bw()
fig12

fig13 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=max_cov_over_sd, y = residuals_values)) +
  geom_smooth(aes(x=max_cov_over_sd, y = residuals_values), method="lm", se = FALSE) +
  xlab("max_cov_over_sd") +
  theme_bw()
fig13

fig14 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=CWRES_each_max, y = residuals_values)) +
  geom_smooth(aes(x=CWRES_each_max, y = residuals_values), method="lm", se = FALSE) +
  xlab("CWRES_each_max") +
  theme_bw()
fig14

fig15 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=CWRES_each_median, y = residuals_values)) +
  geom_smooth(aes(x=CWRES_each_median, y = residuals_values), method="lm", se = FALSE) +
  xlab("CWRES_each_median") +
  theme_bw()
fig15

fig16 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=max_EBE_over_omega_each, y = residuals_values)) +
  geom_smooth(aes(x=max_EBE_over_omega_each, y = residuals_values), method="lm", se = FALSE) +
  xlab("max_EBE_over_omega_each") +
  theme_bw()
fig16

fig17 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=OFVRatio, y = residuals_values)) +
  geom_smooth(aes(x=OFVRatio, y = residuals_values), method="lm", se = FALSE) +
  xlab("OFVRatio") +
  theme_bw()
fig17

fig18 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=OFVRatio2, y = residuals_values)) +
  geom_smooth(aes(x=OFVRatio2, y = residuals_values), method="lm", se = FALSE) +
  xlab("OFVRatio2") +
  theme_bw()
fig18

fig19 <- ggplot(data = predictors_lopinavir) +
  geom_point(aes(x=mean_ETC_omega, y = residuals_values)) +
  geom_smooth(aes(x=mean_ETC_omega, y = residuals_values), method="lm", se = FALSE) +
  xlab("mean_ETC_omega") +
  theme_bw()
fig19




plot_grid(fig12, fig13, fig14, fig15, fig16, fig17, fig18, fig19,
          ncol = 3, nrow = 3)


