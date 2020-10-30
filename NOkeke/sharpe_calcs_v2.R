# Install necessary packages If not done before:
install.packages("lubridate")
install.packages("riskParityPortfolio")
install.packages("NMOF")

#### Initialization, Data Import, log return calcs, and important dates####

# Set Working Directory & necessary packages
setwd("~/Documents/UTO401/Thesis")
library(lubridate) # For date calcs
library(riskParityPortfolio) # For Risk Parity
library(NMOF) # For Min Var

# df will be the dataframe with all raw data
df<-read.csv("US_set.csv")

# As the first col is just numbers, remove, and turn dates into R date format
df<-df[,c(2:length(df))]
df$Date<-as.Date(df$Date,"%Y-%m-%d")

# As log returns are n*m-1 dimensions, we initialize a matrix of that size and perform the calcs for log returns
# log returns = ln(S_n/S_n-1) = ln(S_n)-ln(S_n-1)
dflr<-df[c(2:dim(df)[1]),]
for(i in c(1:(length(df)-1))){
  dflr[,i+1]<-diff(log(df[,i+1]), lag=1)
}

# Option to view output
View(dflr)
rownames(dflr) <- 1:nrow(dflr)

# Find and store the index of the first date available in every month (as it'll be relevant for covariance matrix calcs)

# Floor each value in the date column to the nearest 1st of the month (i.e. for 2001-01-16 this becomes 2001-01-01)
# then subtract each row from the previous row using the diff() function. Since the dates have all been floored, this
# difference will only ≠ 0 (i.e. > 1 or actually > 28) when the index changes from one month to another 
# (i.e. (2001-01-01)-(2001-02-01) = 31 > 1). So "cutoff_dates" is a matrix of 0s and nonzero values at the index where 
# the month changed 
cutoff_dates<-diff(floor_date(dflr$Date, "month"))>1
# Store the indices for each change in month, add 1 because the cutoff_dates vector is shifted back by 1 b/c of the diff() function
cov_dates<-which(cutoff_dates)+1

# Create an array with the indices of each month shifted down 2 spots for the output file. 
# "month3_start_indices" is not used anywhere else in the code except the output file 
month_start_indices <- c(1,1,cov_dates[1:length(cov_dates)-1])

# Add a 1 to cov_dates since the 1st index is the start of the first month in the dataset
cov_dates<-c(1,cov_dates)


#### Computing covariance and portfolio weights ####

# Calculate and store each covariance matrix based on 3 months of daily returns data rolling/shifting monthly
# Note: index starts at "4" - first date where we have enough data
covmatrix<-list()
# length(cov_dates) = number of months in the dataset
for(i in c(4:(length(cov_dates)))){
  # cov_dates[i-3] = index of the 1st day of the 1st month in the 3-month rolling window (i.e. if i = 4, cov_dates[1] = 1)
  # cov_dates[i]-1 = index of the last day of the 3rd month in the 3-month rolling window (i.e. i = 4 cov_dates[4] = index of 
  # first day of 4th month, so cov_dates[4]-1 is the index of the last day of month 3)
  # c(2:dim(dflr)[2]): pull all the columns except 'Date' from dflr 
  covmatrix[[i]]<-cov(dflr[c(cov_dates[i-3]:(cov_dates[i]-1)),c(2:dim(dflr)[2])])
}

# 1 Over N Portfolio:
weights_1n_single_period<-rep(1/(dim(dflr)[2]-1),(dim(dflr)[2]-1))
w_1n<-list()
for(i in c(4:(length(cov_dates)))){
  # store the weights for each time period as the constant value 1/N
  w_1n[[i]]<-weights_1n_single_period
}

# Risk Parity:

w_rp<-list()
for(i in c(4:(length(cov_dates)))){
  # Input the covariance matrix into the riskParityPortfolio function and store the weights 
  w_rp[[i]]<-as.numeric(riskParityPortfolio(covmatrix[[i]])$w)
}

# Min Variance:

w_mv<-list()
for(i in c(4:(length(cov_dates)))){
  # Use the minvar() function to compute min variance portfolio weights 
  w_mv[[i]]<-round(as.numeric(minvar(as.matrix(covmatrix[[i]]), wmin = 0.000000001, wmax = 1, method = "qp")), digits = 5)
}

#### Performance of Portfolios ####

# Initialize vectors to store daily portfolio performance
# Calculating daily portfolio performance because we're using daily returns data 
daily_perform_1n<-rep(NA, nrow(dflr))
daily_perform_rp<-rep(NA, nrow(dflr))
daily_perform_mv<-rep(NA, nrow(dflr))

#Calculate vectors of daily portfolio performance
for(i in c(4:(length(cov_dates)))){
  # Same indexing as for the covariance matrix
  # c(cov_dates[i-1]:(cov_dates[i]-1)) is a vector of the indexes of all the days within a three month period
  # starting at index cov_dates[i-1]
  # dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),c(2:dim(dflr)[2])] = all daily returns data in dflr for month 3 of the 
  # 3-month period that the covariance matrix was calculated on
  daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(
                                                        dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),c(2:dim(dflr)[2])], 
                                                        MARGIN=2, STATS=w_1n[[i]], FUN=`*`))   # MARGIN = 2 indicates sweep by columns
                                                        # STATS = w_1n[[i]] means multiply (defined by FUN='*') each column in dflr by 
                                                        # the corresponding column in the 1/N portfolio weights vector 
  # sweep essentially goes through and multiplies each column of returns by its corresponding portfolio weight
  # then it returns a matrix that's the same size as the original dflr matrix
  # row sums then sums this matrix by row to find total portfolio value on each day
}

for(i in c(4:(length(cov_dates)))){
  # Same formula as above but using risk parity portfolio weights
  daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),
                                                                           c(2:dim(dflr)[2])], MARGIN=2, w_rp[[i]], `*`))
}

for(i in c(4:(length(cov_dates)))){
  # Same formula as above but using mean variance portfolio weights
  daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))]<-rowSums(sweep(dflr[c(cov_dates[i-1]:(cov_dates[i]-1)),
                                                                           c(2:dim(dflr)[2])], MARGIN=2, w_mv[[i]], `*`))
}


# Find summary stats:
# Initialize volatility arrays 
monthly_vol_1n<-rep(NA,length(cov_dates))
monthly_vol_rp<-rep(NA,length(cov_dates))
monthly_vol_mv<-rep(NA,length(cov_dates))
monthly_days_in_month<-rep(NA,length(cov_dates))

# c(cov_dates[i-1]:(cov_dates[i]-1)) vector of indices for the last month (month 3) in each 3-month period

for(i in c(4:(length(cov_dates)))){
  monthly_vol_1n[i]<-sd(daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_vol_rp[i]<-sd(daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_vol_mv[i]<-sd(daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_days_in_month[i]<-length(c(cov_dates[i-1]:(cov_dates[i]-1)))
}

# Returns
monthly_r_1n<-rep(NA,length(cov_dates))
monthly_r_rp<-rep(NA,length(cov_dates))
monthly_r_mv<-rep(NA,length(cov_dates))

# Total portfolio return in a month = sum of the portfolio's daily log return for each day in the month
for(i in c(4:(length(cov_dates)))){
  monthly_r_1n[i]<-sum(daily_perform_1n[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_r_rp[i]<-sum(daily_perform_rp[c(cov_dates[i-1]:(cov_dates[i]-1))])
}
for(i in c(4:(length(cov_dates)))){
  monthly_r_mv[i]<-sum(daily_perform_mv[c(cov_dates[i-1]:(cov_dates[i]-1))])
}

# Sharpe ratio = (monthly return)/[(daily volatility)*sqrt(# of days in a month)]

sharpe_ratio_1n <- monthly_r_1n/(monthly_vol_1n*sqrt(monthly_days_in_month))
sharpe_ratio_rp <- monthly_r_rp/(monthly_vol_rp*sqrt(monthly_days_in_month))
sharpe_ratio_mv <- monthly_r_mv/(monthly_vol_mv*sqrt(monthly_days_in_month))

data.frame()

# Add columns to the sharpes.csv file with the month start and end date
month_start_date<- dflr$Date[month_start_indices]
month_end_date<- dflr$Date[cov_dates]

final_data<-data.frame(month_start_date,month_end_date,monthly_vol_1n, monthly_vol_rp, 
           monthly_vol_mv, monthly_r_1n, monthly_r_rp, monthly_r_mv, monthly_days_in_month, 
           sharpe_ratio_1n,
           sharpe_ratio_rp,
           sharpe_ratio_mv)

setwd("~/Documents/UTO401/Thesis")
write.csv(final_data, file="sharpes.csv")


#Quick plots of Sharpe ratio to see comparative differences in code
par(mar = rep(2, 4))
plot(month_start_date,sharpe_ratio_1n, type="l",col='purple')
lines(dflr$Date[cov_dates],sharpe_ratio_rp,col = "red",type = 'l')
lines(dflr$Date[cov_dates],sharpe_ratio_mv,col = "blue",type = 'l')
