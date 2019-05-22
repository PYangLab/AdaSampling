#' Benchmarking AdaSampling efficacy on noisy labelled data.
#'
#' \code{adaSvmBenchmark()} allows a comparison between the performance
#' of an AdaSampling-enhanced SVM (support vector machine)-
#' classifier against the SVM-classifier on its
#' own. It requires a matrix of features (extracted from a labelled dataset),
#' and two vectors of true labels and labels with noise added as desired.
#' It runs an SVM classifier and returns a matrix which displays the specificity
#' (Sp), sensitivity (Se) and F1 score for each of four conditions:
#' "Original" (classifying with true labels), "Baseline" (classifying with
#' noisy labels), "AdaSingle" (classifying using AdaSampling) and
#' "AdaEnsemble" (classifying using AdaSampling in conjunction with
#' an ensemble of models).
#'
#' AdaSampling is an adaptive sampling-based noise reduction method
#' to deal with noisy class labelled data, which acts as a wrapper for
#' traditional classifiers, such as support vector machines,
#' k-nearest neighbours, logistic regression, and linear discriminant
#' analysis. For more details see \code{?adaSample()}.
#'
#' This function runs evaluates the AdaSampling procedure by adding noise
#' to a labelled dataset, and then running support vector machines on
#' the original and the noisy dataset. Note that this function is for
#' benchmarking AdaSampling performance using what is assumed to be
#' a well-labelled dataset. In order to run AdaSampling on a noisy dataset,
#' please see \code{adaSample()}.
#'
#' @section References:
#' Yang, P., Liu, W., Yang. J. (2017) Positive unlabeled learning via wrapper-based
#' adaptive sampling. \emph{International Joint Conferences on Artificial Intelligence (IJCAI)}, 3272-3279
#'
#' Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018) 
#' AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications. 
#' \emph{IEEE Transactions on Cybernetics}, doi:10.1109/TCYB.2018.2816984
#'
#' @import stats
#'
#' @param data.mat a rectangular matrix or data frame that can be
#' coerced to a matrix, containing the
#' features of the dataset, without class labels. Rownames (possibly
#' containing unique identifiers) will be ignored.
#' @param data.cls a numeric vector containing class labels for the dataset
#'  with added noise.
#' Must be in the same order and of the same length as \code{data.mat}. Labels
#' must be 1 for positive observations, and 0 for negative observations.
#' @param data.cls.truth a numeric vector of true class labels for
#' the dataset. Must be the same order and of the same length as \code{data.mat}. Labels must
#' be 1, for positive observations, and 0 for negative observations.
#' @param cvSeed sets the seed for cross-validation.
#' @param C sets how many times to run the classifier, for the AdaEnsemble
#' condition. See Description above.
#' @param sampleFactor provides a control on the sample size for resampling.
#' @return performance matrix
#' @export
#' @examples
#' # Load the example dataset
#' data(brca)
#' head(brca)
#' 
#' # First, clean up the dataset to transform into the required format.
#' brca.mat <- apply(X = brca[,-10], MARGIN = 2, FUN = as.numeric)
#' brca.cls <- sapply(X = brca$cla, FUN = function(x) {ifelse(x == "malignant", 1, 0)})
#' rownames(brca.mat) <- paste("p", 1:nrow(brca.mat), sep="_")
#' 
#' # Introduce 40% noise to positive class and 30% noise to the negative class
#' set.seed(1)
#' pos <- which(brca.cls == 1)
#' neg <- which(brca.cls == 0)
#' brca.cls.noisy <- brca.cls
#' brca.cls.noisy[sample(pos, floor(length(pos) * 0.4))] <- 0
#' brca.cls.noisy[sample(neg, floor(length(neg) * 0.3))] <- 1
#' 
#' # benchmark classification performance with different approaches
#' \donttest{
#' adaSvmBenchmark(data.mat = brca.mat, data.cls = brca.cls.noisy, data.cls.truth = brca.cls, cvSeed=1)
#' }
#' 
adaSvmBenchmark <- function(data.mat, data.cls, data.cls.truth, cvSeed, C=50, sampleFactor=1){ 
  ##############------------Helper functions--------------##################################
  ### evaluation function
  evaluate <- function(TN, FP, TP, FN, psd=TRUE, print=FALSE) {
    mat <- rbind(TN, FP, TP, FN)

    if (print == TRUE) {
      cat(round(mean(Se(mat)), digits=3))
      cat(" ")

      if (psd==TRUE) {
        cat(round(sd(Se(mat)), digits=3))
        cat(" ")
      }

      cat(round(mean(Sp(mat)), digits=3))
      cat(" ")

      if (psd==TRUE) {
        cat(round(sd(Sp(mat)), digits=3))
        cat(" ")
      }

      cat(round(mean(F1(mat)), digits=3))
      cat(" ")

      if (psd==TRUE) {
        cat(round(sd(F1(mat)), digits=3))
        cat(" ")
      }
    }

    return(c(round(mean(Se(mat)), digits=3),
             round(mean(Sp(mat)), digits=3),
             round(mean(F1(mat)), digits=3)))
  }

  ### Evaluation matrices
  # sensitivity
  Se <- function(mat) {
    apply(mat, 2, function(x) {
      TN <- x[1]
      FP <- x[2]
      TP <- x[3]
      FN <- x[4]
      TP/(TP+FN)
    })
  }

  # specificity
  Sp <- function(mat) {
    apply(mat, 2, function(x) {
      TN <- x[1]
      FP <- x[2]
      TP <- x[3]
      FN <- x[4]
      TN/(FP+TN)
    })
  }

  # F1 score
  F1 <- function(mat) {
    apply(mat, 2, function(x){
      TN <- x[1]
      FP <- x[2]
      TP <- x[3]
      FN <- x[4]
      2*TP/(2*TP+FP+FN)
    })
  }

  
  ##############------------------------------------------##################################
  #Convert to factors
  eval <- matrix(NA, nrow=4, ncol=3)
  colnames(eval) <- c("Se", "Sp", "F1")
  rownames(eval) <- c("Original", "Baseline", "AdaSingle", "AdaEnsemble")
  k <- 5
  set.seed(cvSeed)
  fold <- createFolds(data.cls.truth, k);
  # gold standard (orignal data)
  TP <- TN <- FP <- FN <- c()
  for(i in 1:length(fold)){
    model <- svm(data.mat[-fold[[i]],], data.cls.truth[-fold[[i]]])
    preds <- ifelse(predict(model, data.mat[fold[[i]],])> 0.5, 1, 0)
    TP <- c(TP, sum((data.cls.truth[fold[[i]]] == preds)[data.cls.truth[fold[[i]]] == "1"]))
    TN <- c(TN, sum((data.cls.truth[fold[[i]]] == preds)[data.cls.truth[fold[[i]]] == "0"]))
    FP <- c(FP, sum((data.cls.truth[fold[[i]]] != preds)[preds == "1"]))
    FN <- c(FN, sum((data.cls.truth[fold[[i]]] != preds)[preds == "0"]))
  }
  eval[1,] <- evaluate(TN, FP, TP, FN, psd=FALSE)
  # without correction
  TP <- TN <- FP <- FN <- c()
  for(i in 1:length(fold)){
    model <- svm(data.mat[-fold[[i]],], data.cls[-fold[[i]]])
    preds <- ifelse(predict(model, data.mat[fold[[i]],])> 0.5, 1, 0)
    TP <- c(TP, sum((data.cls.truth[fold[[i]]] == preds)[data.cls.truth[fold[[i]]] == "1"]))
    TN <- c(TN, sum((data.cls.truth[fold[[i]]] == preds)[data.cls.truth[fold[[i]]] == "0"]))
    FP <- c(FP, sum((data.cls.truth[fold[[i]]] != preds)[preds == "1"]))
    FN <- c(FN, sum((data.cls.truth[fold[[i]]] != preds)[preds == "0"]))
  }
  eval[2,] <- evaluate(TN, FP, TP, FN, psd=FALSE)
  
  # single classifier AdaSampling
  TP <- TN <- FP <- FN <- c()
  for (i in 1:length(fold)) {
    Ps <- rownames(data.mat[-fold[[i]],])[which(data.cls[-fold[[i]]] == 1)]
    Ns <- rownames(data.mat[-fold[[i]],])[which(data.cls[-fold[[i]]] == 0)]
    pred <- adaSample(Ps, Ns, data.mat[-fold[[i]],], test.mat=data.mat[fold[[i]],], classifier="svm", sampleFactor)[,"P"]
    TP <- c(TP, sum(pred > 0.5 & data.cls.truth[fold[[i]]] == 1))
    TN <- c(TN, sum(pred < 0.5 & data.cls.truth[fold[[i]]] == 0))
    FP <- c(FP, sum(pred > 0.5 & data.cls.truth[fold[[i]]] == 0))
    FN <- c(FN, sum(pred < 0.5 & data.cls.truth[fold[[i]]] == 1))
  }
  eval[3,] <- evaluate(TN, FP, TP, FN, psd=FALSE)
  
  # ensemble classifier AdaSampling
  TP <- TN <- FP <- FN <- c()
  for (i in 1:length(fold)) {
    Ps <- rownames(data.mat[-fold[[i]],])[which(data.cls[-fold[[i]]] == 1)]
    Ns <- rownames(data.mat[-fold[[i]],])[which(data.cls[-fold[[i]]] == 0)]
    pred <- adaSample(Ps, Ns, data.mat[-fold[[i]],], test.mat=data.mat[fold[[i]],], classifier="svm", C=C, sampleFactor)[,"P"]
    TP <- c(TP, sum(pred > 0.5 & data.cls.truth[fold[[i]]] == 1))
    TN <- c(TN, sum(pred < 0.5 & data.cls.truth[fold[[i]]] == 0))
    FP <- c(FP, sum(pred > 0.5 & data.cls.truth[fold[[i]]] == 0))
    FN <- c(FN, sum(pred < 0.5 & data.cls.truth[fold[[i]]] == 1))
  }
  eval[4,] <- evaluate(TN, FP, TP, FN, psd=FALSE)
  
  return(eval)
}
