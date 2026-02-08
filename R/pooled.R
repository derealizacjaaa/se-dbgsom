# pooled.R - Pooled cross-sectional data preprocessing
#
# Replaces the time series / panel data sliding-window approach.
# A pooled model simply stacks all cross-sectional observations, ignoring
# entity and time structure. This is the standard approach for SOM analysis
# when temporal dynamics are not of interest.

#' Prepare pooled cross-sectional data for DBGSOM training
#'
#' Takes data that may have entity and time identifiers and produces a clean
#' numeric matrix suitable for SOM training, by pooling (stacking) all
#' observations and optionally z-score normalizing.
#'
#' @param data data.frame or matrix of observations
#' @param feature_cols character vector of column names to use as features,
#'   or integer vector of column indices. If NULL, uses all numeric columns.
#' @param entity_col character or integer identifying entity/group column
#'   (will be excluded from features, preserved in metadata). NULL if none.
#' @param time_col character or integer identifying time column
#'   (will be excluded from features, preserved in metadata). NULL if none.
#' @param normalize logical, z-score normalize features (default TRUE)
#' @return list with:
#'   X: numeric matrix (n_samples x n_features), ready for dbgsom
#'   feature_names: character vector of feature names
#'   entity_ids: vector of entity identifiers (or NULL)
#'   time_ids: vector of time identifiers (or NULL)
#'   center: numeric vector of feature means (for denormalization)
#'   scale: numeric vector of feature sds (for denormalization)
#'
#' @examples
#' # Simple numeric matrix
#' pooled <- pooled_prepare(iris[, 1:4])
#' som <- dbgsom(input_dim = ncol(pooled$X))
#' som <- fit(som, pooled$X)
#'
#' # Data frame with entity/time columns
#' pooled <- pooled_prepare(panel_df,
#'                          feature_cols = c("gdp", "inflation", "trade"),
#'                          entity_col = "country",
#'                          time_col = "year")
pooled_prepare <- function(data,
                           feature_cols = NULL,
                           entity_col = NULL,
                           time_col = NULL,
                           normalize = TRUE) {
  data <- as.data.frame(data)

  # Extract entity and time IDs before subsetting
  entity_ids <- NULL
  time_ids <- NULL
  exclude_cols <- character(0)

  if (!is.null(entity_col)) {
    if (is.numeric(entity_col)) entity_col <- names(data)[entity_col]
    entity_ids <- data[[entity_col]]
    exclude_cols <- c(exclude_cols, entity_col)
  }

  if (!is.null(time_col)) {
    if (is.numeric(time_col)) time_col <- names(data)[time_col]
    time_ids <- data[[time_col]]
    exclude_cols <- c(exclude_cols, time_col)
  }

  # Select feature columns
  if (is.null(feature_cols)) {
    # Use all numeric columns except entity/time
    numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
    feature_cols <- setdiff(numeric_cols, exclude_cols)
  } else if (is.numeric(feature_cols)) {
    feature_cols <- names(data)[feature_cols]
  }

  X <- as.matrix(data[, feature_cols, drop = FALSE])
  feature_names <- feature_cols

  # Normalize
  center <- rep(0.0, ncol(X))
  scale_vals <- rep(1.0, ncol(X))
  names(center) <- feature_names
  names(scale_vals) <- feature_names

  if (normalize) {
    for (j in seq_len(ncol(X))) {
      vals <- X[, j]
      vals_clean <- vals[!is.na(vals)]
      if (length(vals_clean) > 1) {
        m <- mean(vals_clean)
        s <- sd(vals_clean)
        if (s > .Machine$double.eps) {
          X[, j] <- (vals - m) / s
          center[j] <- m
          scale_vals[j] <- s
        }
      }
    }
  }

  list(
    X = X,
    feature_names = feature_names,
    entity_ids = entity_ids,
    time_ids = time_ids,
    center = center,
    scale = scale_vals
  )
}
