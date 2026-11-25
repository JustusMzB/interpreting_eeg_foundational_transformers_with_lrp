import glob
import logging
import os
from typing import List, Optional, Dict, Any, Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

class ExperimentResults:
    """
    Utility class to discover and manage experimental results across multiple runs and configurations.
    
    This class assumes a directory structure where each experiment contains multiple 'run_*' folders,
    and optionally an 'evaluation_summary.csv' file aggregating performance metrics.
    """
    def __init__(self, experiment_dir: str):
        """
        Initializes the discovery of runs in the given experiment directory.
        
        Args:
            experiment_dir (str): Root directory of the experiment.
        """
        self.experiment_dir = os.path.abspath(experiment_dir)
        self.summary_path = os.path.join(self.experiment_dir, "evaluation_summary.csv")
        self.performance_df = None
        
        # Load summary performance metrics if available
        if os.path.isfile(self.summary_path):
            self.performance_df = pd.read_csv(self.summary_path)
            # Ensure consistency by sorting by run index
            if 'run' in self.performance_df.columns:
                self.performance_df = self.performance_df.sort_values('run')
        
        self.run_dirs = self._discover_runs()

    def _discover_runs(self) -> List[str]:
        """
        Discovers all run directories matching the expected structure.
        Supports both deep (version_0) and shallow (run_*) structures.
        
        Returns:
            List[str]: Sorted list of absolute paths to run directories.
        """
        # Pattern usually: run_*/task_name/version_0
        run_pattern = os.path.join(self.experiment_dir, "run_*", "*", "version_0")
        dirs = sorted(glob.glob(run_pattern))
        if not dirs:
             # Fallback to shallower pattern
             run_pattern = os.path.join(self.experiment_dir, "run_*")
             dirs = sorted(glob.glob(run_pattern))
        return dirs

    def get_attribution_path(self, run_idx: int, split: str = "test") -> Optional[str]:
        """
        Returns the path to the attribution parquet file or directory for a specific run index.
        
        Args:
            run_idx (int): The 0-based index of the run in discovered directories.
            split (str): Data split name (e.g., 'test', 'val').
            
        Returns:
            Optional[str]: Absolute path to the attribution results, or None if not found.
        """
        if run_idx >= len(self.run_dirs):
            return None
        
        run_dir = self.run_dirs[run_idx]
        # Path can be a single .parquet file or a directory containing multiple .parquet files
        attr_path = os.path.join(run_dir, f"attribution_results_{split}.parquet")
        if os.path.exists(attr_path):
            return attr_path
        return None

class AttributionAnalyzer:
    """
    High-performance analyzer for EEG attributions using Dask for out-of-core computation.
    Supports both legacy (nested lists) and modern (flattened + float16) attribution formats.
    """
    def __init__(self, results: Optional[ExperimentResults] = None):
        """
        Initializes the analyzer with an optional ExperimentResults object.
        """
        self.results = results

    def load_run_data(self, run_idx: int, split: str = "test") -> dd.DataFrame:
        """
        Loads attribution data for a specific run as a Dask DataFrame.
        
        Args:
            run_idx (int): Index of the run to load.
            split (str): Data split name.
            
        Returns:
            dd.DataFrame: Dask DataFrame pointing to the parquet data.
        """
        if not self.results:
            raise ValueError("ExperimentResults not initialized.")
        path = self.results.get_attribution_path(run_idx, split)
        if not path:
            raise FileNotFoundError(f"Attribution file not found for run {run_idx}, split {split}")
        
        # Load using dask (lazy loading)
        ddf = dd.read_parquet(path, engine="pyarrow")
        return ddf

    def load_run_data_to_memory(self, run_idx: int, split: str = "test") -> pd.DataFrame:
        """
        Loads attribution data for a specific run into an in-memory Pandas DataFrame.
        WARNING: This can cause RAM overruns for large datasets.
        
        Args:
            run_idx (int): Index of the run to load.
            split (str): Data split name.
            
        Returns:
            pd.DataFrame: Pandas DataFrame with structured NumPy attributions.
        """
        ddf = self.load_run_data(run_idx, split)
        df = ddf.compute()
        return AttributionAnalyzer._convert_to_numpy(df)

    def load_all_runs_data(
        self, run_indices: Optional[List[int]] = None, split: str = "test"
    ) -> dd.DataFrame:
        """
        Loads attribution data across multiple runs into a single combined Dask DataFrame.
        Adds a 'run_idx' column to identify the source run for each sample.

        Args:
            run_indices (Optional[List[int]]): List of run indices to load.
                                               Defaults to all discovered runs in ExperimentResults.
            split (str): Data split name (e.g. 'test', 'val').

        Returns:
            dd.DataFrame: Combined Dask DataFrame containing data from all specified runs.
        """
        if not self.results:
            raise ValueError("ExperimentResults not initialized.")

        if run_indices is None:
            run_indices = list(range(len(self.results.run_dirs)))

        dfs = []
        for idx in run_indices:
            path = self.results.get_attribution_path(idx, split)
            if not path:
                LOGGER.warning(f"Attribution path for run index {idx} ({split} split) not found. Skipping.")
                continue
            ddf = dd.read_parquet(path, engine="pyarrow")
            ddf["run_idx"] = idx
            dfs.append(ddf)

        if not dfs:
            raise FileNotFoundError(f"No attribution files found for specified run_indices and split '{split}'")

        return dd.concat(dfs, axis=0)

    def calculate_aggregate_statistics(
        self,
        run_indices: Optional[List[int]] = None,
        split: str = "test",
        class_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Calculates attribution statistics aggregated across multiple runs.

        Returns both:
        - 'pooled_classwise': Sample-level classwise statistics aggregated over all samples across all runs.
        - 'per_run_classwise': Dictionary mapping run_idx -> classwise statistics for that run.
        - 'cross_run_aggregates': Mean of run means and inter-run standard deviation for each class and subset.

        Args:
            run_indices (Optional[List[int]]): Indices of runs to aggregate. Defaults to all discovered runs.
            split (str): Data split name.
            class_names (Optional[Dict[int, str]]): Mapping from class IDs to human-readable names.

        Returns:
            Dict[str, Any]: Comprehensive dictionary containing pooled, per-run, and cross-run aggregate statistics.
        """
        if not self.results:
            raise ValueError("ExperimentResults not initialized.")

        if run_indices is None:
            run_indices = list(range(len(self.results.run_dirs)))

        per_run_stats = {}
        for idx in run_indices:
            try:
                ddf_run = self.load_run_data(idx, split=split)
                per_run_stats[idx] = self.calculate_classwise_statistics(ddf_run, class_names=class_names)
            except (FileNotFoundError, ValueError) as e:
                LOGGER.warning(f"Skipping run {idx} during aggregation: {e}")

        if not per_run_stats:
            raise FileNotFoundError(f"No valid run data loaded for aggregation across runs.")

        # Combined pooled statistics
        combined_ddf = self.load_all_runs_data(run_indices=list(per_run_stats.keys()), split=split)
        pooled_classwise = self.calculate_classwise_statistics(combined_ddf, class_names=class_names)

        # Calculate cross-run mean and inter-run std of attribution maps
        cross_run = {}
        first_run_stats = next(iter(per_run_stats.values()))
        labels = list(first_run_stats.keys())

        for label in labels:
            cname = class_names.get(label, str(label)) if class_names else str(label)
            cross_run[label] = {"class_name": cname}

            for subset in ["all", "correct", "incorrect"]:
                key = f"{subset}_mean"
                run_means = [
                    pstats[label][key]
                    for pstats in per_run_stats.values()
                    if label in pstats and key in pstats[label]
                ]
                if run_means:
                    mean_arr = np.mean(run_means, axis=0)
                    std_arr = np.std(run_means, axis=0)
                    cross_run[label][f"{subset}_mean_across_runs"] = mean_arr
                    cross_run[label][f"{subset}_std_across_runs"] = std_arr

        return {
            "pooled_classwise": pooled_classwise,
            "per_run_classwise": per_run_stats,
            "cross_run_aggregates": cross_run,
        }

    @staticmethod
    def _convert_to_numpy(partition: pd.DataFrame) -> pd.DataFrame:
        """
        Helper to convert raw Parquet data in a partition back to structured NumPy arrays.
        
        Handles two formats:
        1. Modern: Flattened 'attribution' column + 'attribution_shape' column.
        2. Legacy: 'attribution' column as nested lists.
        
        Args:
            partition (pd.DataFrame): A single Dask partition (Pandas DataFrame).
            
        Returns:
            pd.DataFrame: Partition with 'attribution' column converted to NumPy arrays.
        """
        if "attribution" in partition.columns:
            if "attribution_shape" in partition.columns:
                # Handle modern optimized format (flattened float16)
                def reshape_attr(row):
                    # Explicitly cast to float16 then reshape to original dimensions
                    attr = np.array(row["attribution"]).astype(np.float16)
                    shape = tuple(row["attribution_shape"])
                    return attr.reshape(shape)
                
                partition["attribution"] = partition.apply(reshape_attr, axis=1)
            else:
                # Handle legacy nested format (float32 lists of lists)
                partition["attribution"] = partition["attribution"].apply(
                    lambda x: np.array([np.array(row) for row in x]) if isinstance(x, (list, np.ndarray)) else x
                )
        return partition

    @staticmethod
    def calculate_statistics(ddf: dd.DataFrame) -> Dict[str, Any]:
        """
        Calculates global mean and standard deviation of attributions across the entire dataset.
        Uses float64 for intermediate sums to maintain numerical stability.

        Args:
            ddf (dd.DataFrame): Dask DataFrame containing attributions.

        Returns:
            Dict[str, Any]: Dictionary mapping prediction classes to their mean/std/count stats.
        """
        # Metadata for map_partitions output
        meta = pd.DataFrame({
            'sum': pd.Series(dtype=object),
            'sum_sq': pd.Series(dtype=object),
            'count': pd.Series(dtype=int)
        }, index=pd.Index([], name='prediction'))

        def sum_and_count(partition):
             if partition.empty:
                 return pd.DataFrame({'sum': [], 'sum_sq': [], 'count': []}, index=pd.Index([], name='prediction'))

             partition = AttributionAnalyzer._convert_to_numpy(partition)

             # Aggregate sums using float64 to avoid overflow
             def stack_sum(x):
                 return np.sum(np.stack(x.values).astype(np.float64), axis=0)

             def stack_sum_sq(x):
                 return np.sum(np.stack(x.values).astype(np.float64)**2, axis=0)

             sums = partition.groupby("prediction")["attribution"].apply(stack_sum)
             sums_sq = partition.groupby("prediction")["attribution"].apply(stack_sum_sq)
             counts = partition.groupby("prediction").size()
             return pd.DataFrame({"sum": sums, "sum_sq": sums_sq, "count": counts})

        # Execute parallel aggregation
        agg = ddf.map_partitions(sum_and_count, meta=meta).compute()

        if agg.empty:
            return {}

        final_stats = {}
        for pred_class in agg.index.unique():
            class_data = agg.loc[pred_class]

            # Handle cases where multiple partitions returned results for the same class
            if isinstance(class_data, pd.Series):
                total_sum = class_data["sum"]
                total_sum_sq = class_data["sum_sq"]
                total_count = class_data["count"]
            else:
                total_sum = np.sum(np.stack(class_data["sum"].values), axis=0)
                total_sum_sq = np.sum(np.stack(class_data["sum_sq"].values), axis=0)
                total_count = class_data["count"].sum()

            mean = total_sum / total_count
            variance = (total_sum_sq / total_count) - (mean**2)
            variance = np.maximum(variance, 0) # Clip negative variance due to precision
            std = np.sqrt(variance)

            final_stats[pred_class] = {
                "mean": mean,
                "std": std,
                "count": total_count
            }

        return final_stats

    def calculate_classwise_statistics(self, ddf: dd.DataFrame, class_names: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
        """
        Calculates statistics for each class, separating correct and incorrect predictions.
        
        Args:
            ddf (dd.DataFrame): Dask DataFrame containing attributions, labels, and predictions.
            class_names (Optional[Dict[int, str]]): Mapping from class IDs to human-readable names.
            
        Returns:
            Dict[str, Any]: Nested dictionary containing class-wise statistics.
        """
        if class_names is None:
            class_names = {}

        def get_subsets_stats(partition):
             if partition.empty:
                 return pd.DataFrame()
             
             partition = AttributionAnalyzer._convert_to_numpy(partition)
             partition["is_correct"] = partition["prediction"] == partition["label"]
             
             results = []
             for label in partition["label"].unique():
                 class_df = partition[partition["label"] == label]
                 
                 for subset in ["all", "correct", "incorrect"]:
                     if subset == "all":
                         sub_df = class_df
                     elif subset == "correct":
                         sub_df = class_df[class_df["is_correct"]]
                     else: # incorrect
                         sub_df = class_df[~class_df["is_correct"]]
                     
                     if sub_df.empty:
                         continue
                         
                     # Convert to float64 for stable aggregation
                     stacked = np.stack(sub_df["attribution"].values).astype(np.float64)
                     results.append({
                         "label": label,
                         "subset": subset,
                         "sum": np.sum(stacked, axis=0),
                         "sum_sq": np.sum(stacked**2, axis=0),
                         "count": len(sub_df)
                     })
             return pd.DataFrame(results)

        meta = pd.DataFrame({
            'label': pd.Series(dtype=int), 
            'subset': pd.Series(dtype=str), 
            'sum': pd.Series(dtype=object), 
            'sum_sq': pd.Series(dtype=object), 
            'count': pd.Series(dtype=int)
        })
        
        agg = ddf.map_partitions(get_subsets_stats, meta=meta).compute()
        
        class_stats = {}
        for (label, subset), group in agg.groupby(["label", "subset"]):
            total_sum = np.sum(np.stack(group["sum"].values), axis=0)
            total_sum_sq = np.sum(np.stack(group["sum_sq"].values), axis=0)
            total_count = group["count"].sum()
            
            if label not in class_stats:
                class_stats[label] = {"class_name": class_names.get(label, str(label))}
            
            mean = total_sum / total_count
            variance = (total_sum_sq / total_count) - (mean**2)
            variance = np.maximum(variance, 0)
            std = np.sqrt(variance)
            
            class_stats[label][f"{subset}_mean"] = mean
            class_stats[label][f"{subset}_std"] = std
            class_stats[label][f"{subset}_count"] = total_count
            
        return class_stats

    def downsample_attributions(self, attr_array: np.ndarray, factor: int = 10) -> np.ndarray:
        """
        Downsamples attributions temporally by averaging over non-overlapping windows.
        
        Args:
            attr_array (np.ndarray): Attribution array of shape (Channels, Time).
            factor (int): Downsampling factor.
            
        Returns:
            np.ndarray: Downsampled attribution array.
        """
        channels, time = attr_array.shape
        new_time = time // factor
        trimmed_attr = attr_array[:, :new_time * factor]
        return trimmed_attr.reshape(channels, new_time, factor).mean(axis=2)

    def map_subjects(self, ddf: dd.DataFrame, dataset: Any) -> dd.DataFrame:
        """
        Joins the Dask DataFrame with subject metadata from the provided dataset.
        
        Args:
            ddf (dd.DataFrame): Dask DataFrame of attributions.
            dataset (Any): Dataset object that supports index-based access and has a 'subject' key.
            
        Returns:
            dd.DataFrame: Dask DataFrame with an added 'subject' column.
        """
        LOGGER.info("Creating subject lookup table...")
        indices = ddf.example_index.unique().compute()
        lookup = pd.DataFrame({
            "example_index": indices,
            "subject": [dataset[int(i)]["subject"] for i in indices]
        })
        
        return ddf.merge(lookup, on="example_index", how="left")

    def spray_clustering(self, ddf: dd.DataFrame, n_clusters: int = 5, downsample_factor: int = 20, n_neighbors: int = 10) -> Dict[str, Any]:
        """
        Simplified Spectral Relevance Analysis (SpRAy) across a Dask DataFrame.
        Identifies distinct decision strategies by clustering attributions.
        
        Args:
            ddf (dd.DataFrame): Dask DataFrame of attributions.
            n_clusters (int): Number of clusters to identify.
            downsample_factor (int): Temporal downsampling factor to reduce feature dimensionality.
            n_neighbors (int): Number of neighbors for SpectralClustering affinity.
            
        Returns:
            Dict[str, Any]: Dictionary containing cluster 'labels' and 'n_clusters'.
        """
        from sklearn.cluster import SpectralClustering
        
        def downsample_partition(partition):
            partition = AttributionAnalyzer._convert_to_numpy(partition)
            # Flatten downsampled attributions for clustering
            return partition["attribution"].apply(lambda x: self.downsample_attributions(x, downsample_factor).flatten())

        LOGGER.info("Downsampling attributions for SpRAy...")
        features = ddf.map_partitions(downsample_partition, meta=(None, 'object')).compute()
        feature_matrix = np.stack(features.values)
        
        # Adjust n_neighbors if too few samples are available
        actual_n_neighbors = min(n_neighbors, len(feature_matrix) - 1)
        
        LOGGER.info(f"Clustering {len(feature_matrix)} examples into {n_clusters} clusters...")
        sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=actual_n_neighbors, n_jobs=-1)
        labels = sc.fit_predict(feature_matrix)
        
        return {"labels": labels, "n_clusters": n_clusters}
