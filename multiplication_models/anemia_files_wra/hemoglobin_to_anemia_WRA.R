########################################
# SUPPLEMENTAL R CODE FOR LSFF 
# MULTIPLICATION MODELS FOR BMGF
# FEB/MARCH 2021
# ALI BOWMAN
#########################################

# clear memory
rm(list=ls())

# disable scientific notation
options(scipen = 999)

# load data from python output
library(readxl)
pacman::p_load(data.table,actuar)
setwd("H:/notebooks/vivarium_research_lsff/multiplication_models/anemia_files_wra")
getwd()
# LOAD BASELINE DATA
mean <- read.csv("mean_hgb_wra.csv")
sd <- read.csv("sd_hgb_wra.csv")
thresholds <- read_excel("thresholds_wra.xlsx")
sb <- read.csv("still.csv")
asfr <- read.csv("asfr.csv")

#Distribution Functions
XMAX = 220
#set ensemble weights
w = c(0.4,0.6)

#----MODEL-------------------------------------------------------------------------------------------------------------

id_vars_mean <- c("location_id", "sex_id", "age_group_id", "year_id","vehicle", "coverage_level")
id_vars <- c("location_id", "sex_id", "age_group_id")

by_vars <- c("location_id", "sex_id", "age_group_id", "draw")

means = as.data.table(mean)
stdev = as.data.table(sd)
stdev[, "X" := NULL]

means.l <- melt(means, id.vars = id_vars_mean, variable.name = "draw", value.name = "mean")
stdev.l <- melt(stdev, id.vars = id_vars, variable.name = "draw", value.name = "stdev")

df <- merge(means.l, stdev.l, by = by_vars)
df[, variance := stdev ^ 2]


### CALCULATE PREGNANCY RATE 
### No uncertainty captured because ASFR covariate doesn't have any 

#Age-spec-preg-prev = (ASFR + stillbirth) * 46/52
setnames(asfr, "mean_value", "asfr")

#stillbirths 
setnames(sb,"mean_value","sbr_mean")
sb <- as.data.table(sb)
sb[,age_group_id := NULL]
sb[,sex_id := NULL]

#Merge - stillbirths are only location-year specific 
df.p <- merge(asfr, sb, by = c("location_id", "year_id"))

#Stillbirth_mean is still births per live birth
df.p <- as.data.table(df.p)
df.p[, prev_pregnant := (asfr + (sbr_mean * asfr)) * 46/52  ]
print(min(df.p$prev_pregnant))
#if(max(df.p$prev_pregnant) > 0.5) stop("PREGNANCY PREV OVER 50% - CHECK MATH?") 


#Subset to pregnant 

#Anemia threshold for <15 does NOT depend on pregnancy!
# comment this out so that we can have pregnancy target prevalence
# for this age group
#df.p <- df.p[age_group_id >= 8]

#Pregnant prev 
preg_prev <- copy(df.p)
preg_prev <- preg_prev[, c(id_vars, "prev_pregnant"), with = FALSE]

#Merge to mean df
df.p <- df.p[, c("location_id", "year_id", "age_group_id", "sex_id", "prev_pregnant"), with = FALSE]
df.p <- merge(df, df.p, by = c("location_id", "year_id", "age_group_id", "sex_id"))

df[, pregnant := 0]
df.p[, pregnant := 1]

#Calc pregnant mean & stdev - sorry for hard coding! This comes from crosswalk in data prep 
df.p[, mean := mean*0.919325]
df.p[, variance := variance * (1.032920188 ^ 2)]
df.p[, stdev := sqrt(variance)]

df.preg <- copy(df.p)
df.preg[, prev_pregnant := NULL]

df <- rbind(df, df.preg)

#MAP THRESHOLDS

df <- merge(df, thresholds, by = c("age_group_id", "sex_id", "pregnant"))


# define relevant constants
EULERS_CONSTANT <- 0.57721566490153286060651209008240243104215933593992
XMAX <- 220
gamma_w <- 0.4
m_gum_w <- 0.6

# define functions from GBD modelers
gamma_mv2p = function(mn, vr){list(shape = mn^2/vr,rate = mn/vr)}

mgumbel_mv2p = function(mn, vr){
  list(
    alpha = XMAX - mn - EULERS_CONSTANT*sqrt(vr)*sqrt(6)/pi,
    scale = sqrt(vr)*sqrt(6)/pi
  ) 
}

# note: pgamma is a standard R function, does not need defining here

pmgumbel = function(q, alpha, scale, lower.tail) 
{ 
  #NOTE: with mirroring, take the other tail
  pgumbel(XMAX-q, alpha, scale, lower.tail=ifelse(lower.tail,FALSE,TRUE)) 
}

###ABBREVIATED - FOR JUST GAMMA AND MGUMBEL
ens_mv2prev <- function(q, mn, vr, w){
  x = q
  
  ##parameters
  params_gamma = gamma_mv2p(mn, vr)
  params_mgumbel = mgumbel_mv2p(mn, vr)
  
  ##weighting
  prev = sum(
    w[1] * pgamma(x, data.matrix(params_gamma$shape),data.matrix(params_gamma$rate)), 
    w[2] * pmgumbel(x,data.matrix(params_mgumbel$alpha),data.matrix(params_mgumbel$scale), lower.tail=T)
  )
  prev
  
}

w = c(0.4,0.6)


### CALCULATE PREVALENCE 
print("CALCULATING MILD")
df[, mild := ens_mv2prev(hgb_upper_mild, mean, variance, w = w) - ens_mv2prev(hgb_lower_mild, mean, variance, w = w)
   , by = 1:nrow(df)]
print("CALCULATING MOD")
df[, moderate := ens_mv2prev(hgb_upper_moderate, mean, variance, w = w) - ens_mv2prev(hgb_lower_moderate, mean, variance, w = w)
   , by = 1:nrow(df)]
print("CALCULATING SEV")
df[, severe := ens_mv2prev(hgb_upper_severe, mean, variance, w = w) - ens_mv2prev(hgb_lower_severe, mean, variance, w = w)
   , by = 1:nrow(df)]
#Anemic is the sum
df[, anemic := mild + moderate + severe]

sevs <- c("mild", "moderate", "severe", "anemic")

#Everything is prevalence (means/stdev entered as continuous)
df[,measure_id := 5]

fwrite(df, "anemia_prev_wra.csv")
