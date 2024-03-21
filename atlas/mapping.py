import os
import sys

import cloudpickle

from typing import Literal, Optional

import numpy as np
import pandas as pd

import scvi
from scvi.model.utils import mde

import scarches
import scanpy as sc

from .wknn import get_wknn, transfer_labels, estimate_presence_score


class AtlasMapper:
    def __init__(self, ref_model):
        self.model_type = self._check_model_type(ref_model)
        self.ref_model = ref_model
        self.ref_adata = ref_model.adata
        self.query_model = None
        self.ref_trans_prob = None

    def map_query(self, query_adata, retrain="partial", **kwargs):
        """
        Map a query dataset to the reference dataset

        Parameters
        ----------
        query_adata : AnnData
            The query dataset to map to the reference dataset
        query_model : str
            The model to use for the query dataset
        retrain : str
            Whether to retrain the query model. Options are "partial", "full" or "none"
        """
        if retrain in ["partial", "full"]:
            if self.model_type == "scanvi":
                self._train_scanvi(query_adata, retrain, **kwargs)
            if self.model_type == "scvi":
                self._train_scvi(query_adata, retrain, **kwargs)
            if self.model_type == "scpoli":
                self._train_scpoli(query_adata, **kwargs)
            self.query_adata = self.query_model.adata
        else:
            self.query_model = self.ref_model
            self.query_adata = query_adata

    def _train_scanvi(self, query_adata, retrain="partial", **kwargs):
        """
        Train a new scanvi model on the query data
        """
        unfrozen = retrain == "full"
        scvi.model.SCANVI.prepare_query_anndata(query_adata, self.ref_model)
        vae_q = scvi.model.SCANVI.load_query_data(
            query_adata, self.ref_model, unfrozen=unfrozen
        )
        vae_q.train(**kwargs)

        self.query_model = vae_q

    def _train_scvi(self, query_adata, query_model, retrain="partial", **kwargs):
        """
        Train a new scvi model on the query data
        """
        unfrozen = retrain == "full"
        scvi.model.SCVI.prepare_query_anndata(query_adata, self.ref_model)
        vae_q = scvi.model.SCVI.load_query_data(
            query_adata, self.ref_model, unfrozen=unfrozen
        )
        vae_q.train(**kwargs)

        self.query_model = vae_q

    def _train_scpoli(self, query_adata, query_model, **kwargs):
        """
        Train a new scpoli model on the query data
        """
        freeze = retrain != "full"
        labeled_indices = np.arange(query_adata.X.shape[0])

        vae_q = scarches.models.scPoli.load_query_data(
            query_adata,
            reference_model=self.ref_model,
            unknown_ct_names=["Unknown"],
            labeled_indices=labeled_indices,
        )

        vae_q.train(**kwargs)

        self.query_model = vae_q

    def _check_model_type(self, model):
        if isinstance(model, scvi.model._scanvi.SCANVI):
            return "scanvi"
        elif isinstance(model, scvi.model._scvi.SCVI):
            return "scvi"
        elif isinstance(model, scarches.models.scpoli.scPoli):
            return "scpoli"
        else:
            raise RuntimeError("This VAE model is currently not supported")

    def _get_latent(self, model, adata, **kwargs):
        if self.model_type in ["scanvi", "scanvi"]:
            return model.get_latent_representation(adata, **kwargs)
        if self.model_type == "scpoli":
            return model.get_latent(adata, **kwargs)

    def compute_wknn(
        self,
        ref_adata=None,
        k: int = 100,
        query2ref: bool = True,
        ref2query: bool = False,
        weighting_scheme: Literal[
            "n", "top_n", "jaccard", "jaccard_square", "gaussian", "dist"
        ] = "jaccard_square",
        top_n: Optional[int] = None,
    ):
        """
        Compute the weighted k-nearest neighbors graph between the reference and query datasets

        Parameters
        ----------
        k : int
            Number of neighbors per cell
        query2ref : bool
            Consider query-to-ref neighbors
        ref2query : bool
            Consider ref-to-query neighbors
        weighting_scheme : str
            How to weight edges in the ref-query neighbor graph
        top_n : int
            The number of top neighbors to consider
        """

        self.ref_adata = ref_adata if ref_adata is not None else self.ref_adata
        ref_latent = self._get_latent(self.query_model, self.ref_adata)
        query_latent = self._get_latent(self.query_model, self.query_adata)

        wknn = get_wknn(
            ref=ref_latent,
            query=query_latent,
            k=k,
            query2ref=query2ref,
            ref2query=ref2query,
            weighting_scheme=weighting_scheme,
            top_n=top_n,
        )

        self.wknn = wknn
        self.ref_adata.obsm["X_latent"] = ref_latent
        self.query_adata.obsm["X_latent"] = query_latent

    def estimate_presence_scores(
        self, split_by=None, random_walk=True, alpha=0.1, n_rounds=100, log=True
    ):
        """
        Estimate the presence score of the query dataset

        Parameters
        ----------
        split_by : str
            The column in the query dataset to split by
        random_walk : bool
            Whether to use random walk to estimate presence score
        alpha : float
            The heat diffusion parameter for the random walk
        n_rounds : int
            The number of rounds for the random walk
        log : bool
            Whether to log the presence score
        """

        scores = estimate_presence_score(
            self.ref_adata,
            self.query_adata,
            self.wknn,
            use_rep_ref_wknn="X_latent",
            use_rep_query_wknn="X_latent",
            ref_trans_prop=self.ref_trans_prob,
            split_by=split_by,
            alpha_random_walk=alpha,
            num_rounds_random_walk=n_rounds,
            log=log,
        )

        self.ref_trans_prob = scores["ref_trans_prop"]
        return scores

    def transfer_labels(self, label_key):
        """
        Transfer labels from the reference dataset to the query dataset

        Parameters
        ----------
        label_key : str
            The column in the reference dataset to transfer
        """

        scores = transfer_labels(
            self.ref_adata,
            self.query_adata,
            self.wknn,
            label_key=label_key,
        )

        return scores

    def save(self, output_dir):
        """
        Save the mapper object to disk

        Parameters
        ----------
        output_dir : str
            The directory to save the mapper object
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "mapper.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(cls, input_dir):
        """
        Load the mapper object from disk

        Parameters
        ----------
        input_dir : str
            The directory to load the mapper object
        """
        with open(os.path.join(input_dir, "mapper.pkl"), "rb") as f:
            mapper = cloudpickle.load(f)
        return mapper
