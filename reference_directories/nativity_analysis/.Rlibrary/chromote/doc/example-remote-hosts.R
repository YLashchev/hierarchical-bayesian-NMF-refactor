## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval=FALSE---------------------------------------------------------------
# library(chromote)
# 
# args <- c(
#   get_chrome_args(),
#   "--headless",
#   "--remote-debugging-address=0.0.0.0",
#   "--remote-debugging-port=9222"
# )
# 
# p <- processx::process$new(find_chrome(), args)
# 
# # To (abruptly) stop this process when you're finished with it:
# p$kill()

