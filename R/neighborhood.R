# neighborhood.R - Gaussian neighborhood kernel and sigma scheduling

#' Gaussian neighborhood function
#'
#' h(d, sigma) = exp(-d^2 / (2 * sigma^2))
#'
#' @param dist numeric grid distance between neurons
#' @param sigma numeric neighborhood width
#' @return numeric neighborhood value
gaussian_neighborhood <- function(dist, sigma) {
  exp(-dist^2 / (2.0 * sigma^2))
}

#' Compute neighborhood width from current grid extent
#'
#' sigma = grid_extent / 2
#'
#' @param neurons named list of neurons
#' @return numeric sigma
compute_sigma <- function(neurons) {
  extent <- grid_extent(neurons)
  max(extent, 1L) / 2.0
}

#' Linear decay of sigma from start to end over n_epochs
#' @param epoch integer current epoch (1-based)
#' @param n_epochs integer total epochs
#' @param sigma_start numeric starting sigma
#' @param sigma_end numeric ending sigma
#' @return numeric current sigma
decay_sigma <- function(epoch, n_epochs, sigma_start, sigma_end) {
  if (n_epochs <= 1L) return(sigma_start)
  progress <- (epoch - 1.0) / (n_epochs - 1.0)
  sigma_start - (sigma_start - sigma_end) * progress
}

#' Compute default sigma start and end values from topology
#' @param neurons named list of neurons
#' @return list with sigma_start and sigma_end
compute_sigma_range <- function(neurons) {
  sigma_start <- compute_sigma(neurons)
  sigma_end <- max(sigma_start * 0.2, 0.5)
  list(sigma_start = sigma_start, sigma_end = sigma_end)
}
