library(car)
data("mtcars")
str(mtcars)
summary(mtcars)
mtcars$am <- as.factor(mtcars$am)
sum(is.na(mtcars))
levels(mtcars$am) <- c("Automatic","Manual")
par(mfrow = c(1,2))

#histogram plotting#
x <- mtcars$mpg
h <- hist(x, breaks = 10,col="red",xlab = "Miles per Gallon",
main="Histogram of Miles per Gallon")
xfit<-seq(min(x),max(x),length=40)
yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
yfit<-yfit*diff(h$mids[1:2])*length(x)
lines(xfit,yfit,col="blue",lwd=2)

#kernel density plot
d <- density(mtcars$mpg)
plot(d,xlab = "MPG",main = "Density Plot of MPG")

#Box Plot
boxplot(mpg~am,data = mtcars,
        col = c("dark grey","Light grey"),
        xlab = "Transmission",
        ylab = "Miles per Gallon",
        main = "MPG by Transmission Type")

#Hypothesis Testing
aggregate(mpg~am, data = mtcars, mean)

autoData <- mtcars[mtcars$am == "Automatic",]
manualData <- mtcars[mtcars$am == "Manual",]
t.test(autoData$mpg, manualData$mpg)


data("mtcars")
sort(cor(mtcars)[1,]) #Correlation

linearmodel <- lm(mpg~am,data = mtcars)
summary(linearmodel)

multimodel <- lm(mpg~am + wt + hp,data = mtcars)
Anova(linearmodel,multimodel)

par(mfrow = c(2,2))
#plotting the models
plot(linearmodel)
plot(multimodel)





 