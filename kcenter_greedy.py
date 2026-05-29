
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances
from protoflow_module.sampling_methods.sampling_def import SamplingMethod


class kCenterGreedy(SamplingMethod):

  def __init__(self, X, y, seed, metric='euclidean'):
    self.X = X
    self.y = y
    self.flat_X = self.flatten_X()
    self.name = 'kcenter'
    self.features = self.flat_X
    self.metric = metric
    self.min_distances = None
    self.n_obs = self.X.shape[0]
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    if reset_dist:
      self.min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = self.features[cluster_centers]
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, model, already_selected, N, **kwargs):
    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      print('Getting transformed features...')
      self.features = model.transform(self.X)
      print('Calculating distances...')
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    for _ in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))


    self.already_selected = already_selected

    return new_batch

