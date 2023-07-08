# Visualize results of epsilon non-dominated sorting 
# Elle Stark July 2023

library(ggplot2)
library(tidyverse)

setwd('C:/Users/elles/Documents/CU_Boulder/Research/Borg_processing_code/borgRW-parser/src/borg_parser/data')
percent_data <- read.csv('nondom_percents.csv')
norm_percent_data <- percent_data %>%
  mutate(Original.eps=Original.eps/All.Policies-1) %>%
  mutate(Double.eps=Double.eps/All.Policies-1) %>%
  mutate(Tenx.eps=Tenx.eps/All.Policies-1) %>%
  mutate(All.Policies=All.Policies/All.Policies-1)

percent_data_long <- pivot_longer(percent_data, cols = 2:5, names_to = 'Epsilon', values_to = 'Percent')
norm_data_long <- pivot_longer(norm_percent_data, cols = 2:5, names_to = 'Epsilon', values_to = 'Percent')

raw_percent_plot <- ggplot(percent_data_long, 
                           aes(x=factor(Epsilon, level=c('All.Policies', 'Original.eps', 'Double.eps', 'Tenx.eps')), 
                               y=Percent, col=Run)) +
  geom_point() +
  geom_path(aes(group=Run)) +
  xlab('Run - Finest to coarsest epsilons from L to R') +
  ylab('Percent Contributed to Combined Set') +
  theme_bw()

raw_percent_plot

norm_percent_plot <- ggplot(norm_data_long, 
                            aes(x=factor(Epsilon, level=c('All.Policies', 'Original.eps', 'Double.eps', 'Tenx.eps')), 
                                y=Percent, col=Run)) +
  geom_point() +
  geom_path(aes(group=Run)) +
  ylab('Change in Percent Contributed to Combined Set') +
  xlab('Run - Finest to coarsest epsilons from L to R') +
  theme_bw()

norm_percent_plot
