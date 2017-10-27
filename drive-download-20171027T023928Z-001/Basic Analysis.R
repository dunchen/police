install.packages("readr")
install.packages(c("ggplot2", "devtools", "dplyr", "stringr"))
install.packages(c("maps", "mapdata"))
rm(list=ls())
##############Preparation###############
library(ggplot2)
library(ggmap)
library(maps)
library(mapdata)
library(readr)

getwd()
setwd("/Users/Ruby/R Projects/PoliceDataChallenge")
main <- read_csv("PDI_Police_Calls_For_Service__CAD_.csv")
sample = main[1:10000,]
head(main)
View(main)
head(sample)
######################Map####################
cbbox<- make_bbox(lat = sample$LATITUDE_X, lon = sample$LONGITUDE_X, f = .1)
cbbox
cin_map <- get_map(location = cbbox, maptype = "terrain", source = "google", zoom=15)
ggmap(cin_map) + geom_point(data = sample, mapping = aes(x = LONGITUDE_X, y = LATITUDE_X), color = sample$PRIORITY_COLOR)
