#' Implementation of a feature weighted k-nearest neighbour classifier.
#' 
#' @param train.mat training data matrix, without class labels.
#' @param test.mat test data matrix, without class labels.
#' @param cl class labels for training data.
#' @param k number of nearest neighbour to be used.
#' @param weights weights to be assigned to each feautre.
#' 
#' @export
#' 
weightedKNN <- function(train.mat, test.mat, cl, k=3, weights){
  #Calculate cross-distance matrix     
  Ds <- (train.mat^2)%*%weights%*%t(rep(1,nrow(test.mat))) + t((test.mat^2)%*%weights%*%t(rep(1,nrow(train.mat)))) - 2*(train.mat*(rep(1,nrow(train.mat)))%*%t(sqrt(weights)))%*%t(test.mat*(rep(1,nrow(test.mat)))%*%t(sqrt(weights)))
  
  #Calculate prediction
  u <- sort(unique(cl))
  preds <- t(apply(Ds, 2, function(x)table(cl[order(x)][1:k])[as.character(u)]/k))
  
  colnames(preds) = u
  return(preds)
}
