import numpy as np
from scipy import stats
from typing import Any

class Template:
    def __init__(self, mean: np.ndarray, covariance: np.ndarray):
        self.distribution = stats.multivariate_normal(mean, covariance)

    def predict(self, feature_vector) -> float:
        return self.distribution.logpdf(feature_vector)

def make_template(features: np.ndarray) -> Template:
    mean = np.mean(features, axis=0)
    covariance = np.cov(features.T)
    return Template(mean, covariance)

def make_template_foreach_label(features: np.ndarray, labels):#: np.ndarray[Any]): #-> dict[Any, Template]:
    # Sort the feature vectors into buckets according to their label
    buckets = {label: [] for label in np.unique(labels)}
    for feature_vector, label in zip(features, labels):
        buckets[label].append(feature_vector)

    # Make a template for each label
    templates = {label: make_template(np.array(features)) for label, features in buckets.items()}
    return templates

def find_best_poi(traces: np.ndarray, labels: np.ndarray, num_poi: int, poi_spacing=10):
    _, num_samples = traces.shape
    
    # Compute the average trace for each label
    buckets_sum_and_len = {label: (0, 0) for label in np.unique(labels)}
    for trace, label in zip(traces, labels):
        bucket_sum, bucket_len = buckets_sum_and_len[label]
        buckets_sum_and_len[label] = (bucket_sum + trace, bucket_len + 1)
    bucket_avg = {label: bucket_sum / bucket_len for label, (bucket_sum, bucket_len) in buckets_sum_and_len.items()}

    # Compute the sum of differences in order to find points of interest
    sum_of_diffs = np.zeros(num_samples)
    for avg_i in bucket_avg.values():
        for avg_j in bucket_avg.values():
            sum_of_diffs += np.abs(avg_i - avg_j)

    # Select the points of interest
    poi_indices = []
    for _ in range(num_poi):
        poi_index = np.argmax(sum_of_diffs)
        poi_indices.append(poi_index)
        sum_of_diffs[max(0, poi_index - poi_spacing) : min(poi_index + poi_spacing, len(sum_of_diffs))] = -np.inf

    return np.sort(poi_indices)
