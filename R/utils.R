# utils.R - Utility functions for dbgsom package
#
# Position keys: neurons are stored in a named list keyed by "x,y" strings.
# Positions are represented as integer vectors c(x, y).

#' Create a position key string from x, y coordinates
#' @param x integer x coordinate
#' @param y integer y coordinate
#' @return character key "x,y"
pos_key <- function(x, y) {
  paste0(x, ",", y)
}

#' Create a position key from a position vector
#' @param pos integer vector c(x, y)
#' @return character key "x,y"
pos_to_key <- function(pos) {
  paste0(pos[1], ",", pos[2])
}

#' Parse a position key back to integer vector
#' @param key character key "x,y"
#' @return integer vector c(x, y)
key_to_pos <- function(key) {
  as.integer(strsplit(key, ",", fixed = TRUE)[[1]])
}

#' Create a new neuron
#' @param weights numeric vector of weights
#' @param error accumulated quantization error
#' @param hits number of BMU selections
#' @return list with weights, error, hits
new_neuron <- function(weights, error = 0.0, hits = 0L) {
  list(weights = weights, error = error, hits = hits)
}
