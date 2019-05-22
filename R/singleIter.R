#' \code{singleIter()} applies a single iteraction of AdaSampling procedure. It
#' returns the probabilities of all samples as being a positive (P) or negative
#' (N) instance, as a two column data frame.
#'
#' Classification algorithms included are support vector machines (svm),
#' k-nearest neighbours (knn), logistic regression (logit), linear discriminant
#' analysis (lda), feature weighted knn (wKNN).

#' @section References:
#' Yang, P., Liu, W., Yang. J. (2017) Positive unlabeled learning via wrapper-based
#' adaptive sampling. \emph{International Joint Conferences on Artificial Intelligence (IJCAI)}, 3272-3279
#'
#' Yang, P., Ormerod, J., Liu, W., Ma, C., Zomaya, A., Yang, J.(2018) 
#' AdaSampling for positive-unlabeled and label noise learning with bioinformatics applications. 
#' \emph{IEEE Transactions on Cybernetics}, doi:10.1109/TCYB.2018.2816984
#'
#' @import class
#' @import e1071
#' @import MASS
#' @import caret
#'
#' @param Ps names (name as index) of positive examples
#' @param Ns names (name as index) of negative examples
#' @param dat training data matrix, without class labels.
#' @param test test data matrix, without class labels.
#' Training data matrix will be used for testing if this is NULL (default).
#' @param pos.probs a numeric vector of containing probability of positive examples been positive
#' @param una.probs a numeric vector of containing probability of negative or unannotated examples been negative
#' @param classifier classification algorithm to be used for learning. Current options are
#' support vector machine, \code{"svm"}, k-nearest neighbour, \code{"knn"}, logistic regression \code{"logit"},
#' linear discriminant analysis \code{"lda"}, and feature weighted knn, \code{"wKNN"}.
#' @param sampleFactor provides a control on the sample size for resampling.
#' @param seed sets the seed.
#' @param weights feature weights, required when using weighted knn.
#' @export

singleIter <- function(Ps, Ns, dat, test=NULL, pos.probs=NULL, una.probs=NULL, classifier="svm", sampleFactor, seed, weights) {
  set.seed(seed);
  
  positive.train <- c()
  positive.cls <- c()
  
  # determine the proper sample size for creating a balanced dataset
  sampleN <- ifelse(length(Ps) < length(Ns), length(Ps), length(Ns))
  
  # bootstrap sampling to build the positive training set (labeled as 'P')
  idx.pl <- unique(sample(x=Ps, size=sampleFactor*sampleN, replace=TRUE, prob=pos.probs[Ps]))
  positive.train <- dat[idx.pl,]
  positive.cls <- rep("P", nrow(positive.train))
  
  # bootstrap sampling to build the "unannotate" or "negative" training set (labeled as 'N')
  idx.dl <- unique(sample(x=Ns, size=sampleFactor*sampleN, replace=TRUE, prob=una.probs[Ns]))
  unannotate.train <- dat[idx.dl,]
  unannotate.cls <- rep("N", nrow(unannotate.train))
  
  # combine data
  train.sample <- rbind(positive.train, unannotate.train)
  rownames(train.sample) <- NULL;
  cls <- as.factor(c(positive.cls, unannotate.cls))
  
  # training svm classifier
  if (classifier == "svm") {
    model.svm <- svm(train.sample, cls, probability=TRUE, scale=TRUE);
    svm.pred <- c();
    if (is.null(test)) {
      svm.pred <- predict(model.svm, dat, decision.values=TRUE, probability=TRUE);
    } else {
      svm.pred <- predict(model.svm, test, decision.values=TRUE, probability=TRUE);
    }
    return(attr(svm.pred,"probabilities"));
    
  } else if (classifier == "knn") {
    # training knn classifier
    if (is.null(test)) {
      knn.fit <- knn(train.sample, dat, cl=cls, k=5, prob=TRUE)
      
      p <- attr(knn.fit, "prob")
      idx <- which(knn.fit == "N")
      p[idx] <- 1- p[idx]
      knn.pred <- cbind(p, 1 - p)
      colnames(knn.pred) <- c("P", "N")
      rownames(knn.pred) <- rownames(dat)
      return(knn.pred)
    } else {
      test.mat <- test
      rownames(test.mat) <- NULL
      knn.fit <- knn(train.sample, test.mat, cl=cls, k=5, prob=TRUE)
      
      p <- attr(knn.fit, "prob")
      idx <- which(knn.fit == "N")
      p[idx] <- 1- p[idx]
      knn.pred <- cbind(p, 1 - p)
      colnames(knn.pred) <- c("P", "N")
      rownames(knn.pred) <- rownames(test)
      return(knn.pred)
    }
  } else if (classifier == "logit") {
    logit.model <- glm(cls~., family=binomial(link='logit'), data=data.frame(train.sample, cls))
    if (is.null(test)) {
      p <- predict(logit.model, newdata=data.frame(dat), type='response')
      logit.pred <- cbind(p, 1-p)
      colnames(logit.pred) <- c("P", "N")
      rownames(logit.pred) <- rownames(dat)
      return(logit.pred)
    } else {
      test.mat <- data.frame(test)
      rownames(test.mat) <- NULL
      colnames(test.mat) <- colnames(dat)
      p <- predict(logit.model, newdata=test.mat, type='response')
      logit.pred <- cbind(p, 1-p)
      colnames(logit.pred) <- c("P", "N")
      rownames(logit.pred) <- rownames(test)
      return(logit.pred)
    }
  } else if (classifier == "lda") {
    lda.model <- MASS::lda(cls~., data=data.frame(train.sample, cls))
    if (is.null(test)) {
      lda.pred <- predict(lda.model, data.frame(dat))$posterior
      colnames(lda.pred) <- c("N", "P")
      rownames(lda.pred) <- rownames(dat)
      return(lda.pred)
    } else {
      test.mat <- data.frame(test)
      rownames(test.mat) <- NULL
      colnames(test.mat) <- colnames(dat)
      lda.pred <- predict(lda.model, test.mat)$posterior
      colnames(lda.pred) <- c("N", "P")
      rownames(lda.pred) <- rownames(test)
      return(lda.pred)
    }
  } else if (classifier == "wKNN") {
    # training a modified knn classifier
    if (is.null(weights)) {
      stop("need to specify weights for using weighted knn!");
    }
    if (is.null(test)) {
      wKNN.pred <- weightedKNN(train.sample, dat, cl=cls, k=3, weights)
      rownames(wKNN.pred) <- rownames(dat)
      return(wKNN.pred)
    } else {
      test.mat <- test
      wKNN.pred <- weightedKNN(train.sample, test.mat, cl=cls, k=3, weights)
      rownames(wKNN.pred) <- rownames(test)
      return(wKNN.pred)
    }
  }
}
