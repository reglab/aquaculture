#
# Estimates an upper bound on the population of cages in the No prediction, not near known facility stratum.
# 

library(ggplot2)

# Stratum parameters (number of images, number of samples)
I_6 <- 783355
S_6 <- 10518

# Number of simulations for each proportion
K <- 10000

# For each proportion, we compute whether with 50% probability we can find a sample including zero labels.
compute <- function(r) {
  labels <- c()
  for (k in 1:K) {
    samples <- rbinom(n=S_6, size=1, prob=r)
    num_labels <- sum(samples)
    labels <- c(labels, num_labels)
  }

  all_zeros_50 <- sort(labels)[as.integer(K / 2)] # 50% probability of a sample of all zeros
  return(list(all_zeros_50=all_zeros_50))
}


rate_df <- data.frame()
for (rate in seq(0.00001, 0.0001, 0.00001)){
  output <- compute(r=rate)
  rate_df <- rbind(
    rate_df, 
    data.frame(rate=rate,
               all_zeros_50=output$all_zeros_50))
}

ggplot(data=rate_df, aes(x=rate, y=all_zeros_50)) + geom_point() 

# Population computation
# * Smallest rate in rate_df for which we find a non-zero cage number
final_r <- 7e-05 
num_images_with_cages <- round(final_r * I_6, 0)
# * Average number of cages per image in our PredictionFacility
cages_per_image <- 5 
pop_estimate_stratum <- num_images_with_cages * cages_per_image
# * Number of CF labels in strata I1 - I5 
# (see ContributionDataset.py or Stratum table in draft)
num_cages_I_1_5 <- 4010 
pop_upper_bound_total <- pop_estimate_stratum + num_cages_I_1_5
