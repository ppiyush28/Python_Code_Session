library(shiny)
library(datasets)
Data <- mtcars

ui <- fluidPage(
  titlePanel("Need For Average"),align="Center",
  
  mainPanel( 
    plotOutput("mpgPlot"),
    plotOutput("histogram"),
    plotOutput("density")
  )
)
server <- function(input, output) {
  output$mpgPlot <- renderPlot({
            boxplot(mpg~am,data = mtcars,
            col = c("dark green","Light green"),
            xlab = "Transmission",
            ylab = "Miles per Gallon",
            main = "MPG by Transmission Type")
  })
  output$histogram <- renderPlot({
    x <- mtcars$mpg
    h <- hist(x, breaks = 10,col="red",xlab = "Miles per Gallon",
              main="Histogram of Miles per Gallon")
    xfit<-seq(min(x),max(x),length=40)
    yfit<-dnorm(xfit,mean=mean(x),sd=sd(x))
    yfit<-yfit*diff(h$mids[1:2])*length(x)
    lines(xfit,yfit,col="blue",lwd=2)
  })
  output$density <- renderPlot({
    d <- density(mtcars$mpg)
    plot(d,xlab = "MPG",main = "Density Plot of MPG")
    
  })
 # output$linearmodel <- renderPrint({
 #   linearmodel <- lm(mpg~am,data = mtcars)
 #   par(mfrow = c(2,2))
  #  plot(linearmodel)
    
 # })
 # output$multimodel <- renderPrint({
 #   multimodel <- lm(mpg~am + wt + hp,data = mtcars)
 #   par(mfrow = c(2,2))
  #  plot(multimodel)
    
#  })
  
  
  
  
  
  
  
  
  
  
  
  
  
}
shinyApp(ui = ui, server = server)