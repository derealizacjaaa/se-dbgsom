# topology.R - Hexagonal grid topology for DBGSOM
#
# Implements odd-r offset hexagonal coordinate system.
# Neurons are stored in a named list keyed by "x,y" strings.

#' Get 6-connected neighbor positions for hexagonal grid (odd-r offset)
#'
#' In odd-r offset, odd rows are shifted right by 0.5.
#'
#' @param pos integer vector c(x, y)
#' @return list of integer vectors, each c(x, y)
get_neighbors_hex <- function(pos) {
  x <- pos[1]
  y <- pos[2]

  if (y %% 2 == 0) {
    # Even row
    list(
      c(x + 1L, y),      # East
      c(x,     y - 1L),   # North-East
      c(x - 1L, y - 1L),  # North-West
      c(x - 1L, y),       # West
      c(x - 1L, y + 1L),  # South-West
      c(x,     y + 1L)    # South-East
    )
  } else {
    # Odd row (shifted right)
    list(
      c(x + 1L, y),      # East
      c(x + 1L, y - 1L),  # North-East
      c(x,     y - 1L),   # North-West
      c(x - 1L, y),       # West
      c(x,     y + 1L),   # South-West
      c(x + 1L, y + 1L)   # South-East
    )
  }
}

#' Hexagonal grid distance using cube coordinate conversion
#'
#' Converts odd-r offset to cube coordinates and computes max of axis differences.
#'
#' @param pos1 integer vector c(x, y)
#' @param pos2 integer vector c(x, y)
#' @return integer grid distance
grid_distance_hex <- function(pos1, pos2) {
  # Convert odd-r offset to cube coordinates
  to_cube <- function(x, y) {
    cx <- x - (y - bitwAnd(y, 1L)) %/% 2L
    cz <- y
    cy <- -cx - cz
    c(cx, cy, cz)
  }
  c1 <- to_cube(pos1[1], pos1[2])
  c2 <- to_cube(pos2[1], pos2[2])
  max(abs(c1[1] - c2[1]), abs(c1[2] - c2[2]), abs(c1[3] - c2[3]))
}

#' Check if a neuron position is on the boundary
#'
#' A position is boundary if any of its 6 hex neighbors is empty.
#'
#' @param neurons named list of neurons (keyed by "x,y")
#' @param pos integer vector c(x, y)
#' @return logical
is_boundary <- function(neurons, pos) {
  nbrs <- get_neighbors_hex(pos)
  for (nbr in nbrs) {
    if (is.null(neurons[[pos_to_key(nbr)]])) {
      return(TRUE)
    }
  }
  FALSE
}

#' Get empty neighbor positions
#' @param neurons named list of neurons
#' @param pos integer vector c(x, y)
#' @return list of position vectors for empty neighbor slots
get_empty_neighbors <- function(neurons, pos) {
  nbrs <- get_neighbors_hex(pos)
  empty <- list()
  for (nbr in nbrs) {
    if (is.null(neurons[[pos_to_key(nbr)]])) {
      empty <- c(empty, list(nbr))
    }
  }
  empty
}

#' Get occupied neighbor positions
#' @param neurons named list of neurons
#' @param pos integer vector c(x, y)
#' @return list of position vectors for occupied neighbor slots
get_occupied_neighbors <- function(neurons, pos) {
  nbrs <- get_neighbors_hex(pos)
  occupied <- list()
  for (nbr in nbrs) {
    if (!is.null(neurons[[pos_to_key(nbr)]])) {
      occupied <- c(occupied, list(nbr))
    }
  }
  occupied
}

#' Get occupied neighbors at exactly a specified grid distance
#' @param neurons named list of neurons
#' @param pos integer vector c(x, y)
#' @param dist integer target distance
#' @return list of position vectors
get_neighbors_at_distance <- function(neurons, pos, dist) {
  result <- list()
  for (key in names(neurons)) {
    other_pos <- key_to_pos(key)
    if (grid_distance_hex(pos, other_pos) == dist) {
      result <- c(result, list(other_pos))
    }
  }
  result
}

#' Get grid bounding box
#' @param neurons named list of neurons
#' @return list with x_min, x_max, y_min, y_max
grid_bounds <- function(neurons) {
  if (length(neurons) == 0) {
    return(list(x_min = 0L, x_max = 0L, y_min = 0L, y_max = 0L))
  }
  all_pos <- lapply(names(neurons), key_to_pos)
  xs <- vapply(all_pos, `[`, integer(1), 1)
  ys <- vapply(all_pos, `[`, integer(1), 2)
  list(x_min = min(xs), x_max = max(xs), y_min = min(ys), y_max = max(ys))
}

#' Get maximum extent of the grid
#' @param neurons named list of neurons
#' @return integer max of width and height
grid_extent <- function(neurons) {
  b <- grid_bounds(neurons)
  max(b$x_max - b$x_min + 1L, b$y_max - b$y_min + 1L)
}

#' Initialize a hexagonal grid of neurons with random weights
#' @param input_dim integer dimension of weight vectors
#' @param init_rows integer number of rows
#' @param init_cols integer number of columns
#' @return named list of neurons
init_topology <- function(input_dim, init_rows = 2L, init_cols = 2L) {
  neurons <- list()
  for (y in 0:(init_cols - 1L)) {
    for (x in 0:(init_rows - 1L)) {
      key <- pos_key(x, y)
      neurons[[key]] <- new_neuron(rnorm(input_dim))
    }
  }
  neurons
}

#' Get weight matrix from neurons (input_dim x n_neurons)
#' @param neurons named list of neurons
#' @return list with pos_keys (character vector) and W (matrix: input_dim x n_neurons)
get_weights_matrix <- function(neurons) {
  keys <- names(neurons)
  n <- length(keys)
  if (n == 0) return(list(pos_keys = character(0), W = matrix(nrow = 0, ncol = 0)))
  input_dim <- length(neurons[[1]]$weights)
  W <- matrix(NA_real_, nrow = input_dim, ncol = n)
  for (i in seq_along(keys)) {
    W[, i] <- neurons[[keys[i]]]$weights
  }
  list(pos_keys = keys, W = W)
}
