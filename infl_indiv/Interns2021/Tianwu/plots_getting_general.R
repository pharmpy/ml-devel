library(readr)
library(ggplot2)
library(cowplot)
predictors_data_copy <- read_csv("predictors_drug.csv") # chage this the dataset we want to plot 

# switch the nonliear or linear that you want to plot
predictors_data_copy <- predictors_data_copy[predictors_data_copy$lin_model == 0, ] # for nonlinear
##  OR 
predictors_data_copy <- predictors_data_copy[predictors_data_copy$lin_model == 1, ] # for linear

### 1st plot: dofv vs predictors
fig4 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Obsi_Obs_Subj, y = dofv)) +
  geom_smooth(aes(x=Obsi_Obs_Subj, y = dofv), method="lm", se = FALSE) +
  xlab("nobsi_over_avg") +
  theme_bw()

fig5 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_cov, y = dofv)) +
  geom_smooth(aes(x=Max_cov, y = dofv), method="lm", se = FALSE) +
  xlab("max_cov_over_sd") +
  theme_bw()

fig6 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_CWRESi, y = dofv)) +
  geom_smooth(aes(x=Max_CWRESi, y = dofv), method="lm", se = FALSE) +
  xlab("CWRES_each_max") +
  theme_bw()

fig7 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Median_CWRESi, y = dofv)) +
  geom_smooth(aes(x=Median_CWRESi, y = dofv), method="lm", se = FALSE) +
  xlab("CWRES_each_median") +
  theme_bw()

fig8 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_EBEij_omegaj, y = dofv)) +
  geom_smooth(aes(x=Max_EBEij_omegaj, y = dofv), method="lm", se = FALSE) +
  xlab("max_EBE_over_omega_each") +
  theme_bw()

fig9 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=OFVRatio, y = dofv)) +
  geom_smooth(aes(x=OFVRatio, y = dofv), method="lm", se = FALSE) +
  xlab("OFVRatio") +
  theme_bw()

fig10 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=mean_ETC_omega, y = dofv)) +
  geom_smooth(aes(x=mean_ETC_omega, y = dofv), method="lm", se = FALSE) +
  xlab("mean_ETC_omega") +
  theme_bw()


plot_grid(fig4, fig5, fig6, fig7, fig8, fig9, fig10 , 
          ncol = 3, nrow = 3)


### 2nd plot: residual vs predictors

fig4 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Obsi_Obs_Subj, y = residual)) +
  geom_smooth(aes(x=Obsi_Obs_Subj, y = residual), method="lm", se = FALSE) +
  xlab("nobsi_over_avg") +
  theme_bw()

fig5 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_cov, y = residual)) +
  geom_smooth(aes(x=Max_cov, y = residual), method="lm", se = FALSE) +
  xlab("max_cov_over_sd") +
  theme_bw()

fig6 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_CWRESi, y = residual)) +
  geom_smooth(aes(x=Max_CWRESi, y = residual), method="lm", se = FALSE) +
  xlab("CWRES_each_max") +
  theme_bw()

fig7 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Median_CWRESi, y = residual)) +
  geom_smooth(aes(x=Median_CWRESi, y = residual), method="lm", se = FALSE) +
  xlab("CWRES_each_median") +
  theme_bw()

fig8 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=Max_EBEij_omegaj, y = residual)) +
  geom_smooth(aes(x=Max_EBEij_omegaj, y = residual), method="lm", se = FALSE) +
  xlab("max_EBE_over_omega_each") +
  theme_bw()

fig9 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=OFVRatio, y = residual)) +
  geom_smooth(aes(x=OFVRatio, y = residual), method="lm", se = FALSE) +
  xlab("OFVRatio") +
  theme_bw()

fig10 <- ggplot(data = predictors_data_copy) +
  geom_point(aes(x=mean_ETC_omega, y = residual)) +
  geom_smooth(aes(x=mean_ETC_omega, y = residual), method="lm", se = FALSE) +
  xlab("mean_ETC_omega") +
  theme_bw()


plot_grid(fig4, fig5, fig6, fig7, fig8, fig9, fig10 , 
          ncol = 3, nrow = 3)
