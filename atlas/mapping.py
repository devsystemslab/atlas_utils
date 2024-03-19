import os
import sys

from typing import Literal, Optional

import numpy as np
import pandas as pd

import scvi
from scvi.model.utils import mde

import scanpy as sc

from .wknn import get_wknn


class AtlasMapper:
    def __init__(self, ref_model, model="scanvi"):
        self.ref_model = ref_model
        self.adata = ref_model
        self.query_model = None

    def map_query(self, query_adata, query_model, retrain="partial", **kwargs):
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
            if self.model == "scanvi":
                self._train_scanvi(query_adata, query_model, retrain, **kwargs)
            if self.model == "scvi":
                self._train_scvi(query_adata, query_model, retrain, **kwargs)
            if self.model == "scpoli":
                self._train_scpoli(query_adata, query_model, **kwargs)
        else:
            self.query_model = self.ref_model

    def _train_scanvi(self, query_adata, query_model, retrain="partial", **kwargs):
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

    def compute_wknn(
        ref,
        query,
        ref2=None,
        k: int = 100,
        query2ref: bool = True,
        ref2query: bool = True,
        weighting_scheme: Literal[
            "n", "top_n", "jaccard", "jaccard_square"
        ] = "jaccard_square",
        top_n: Optional[int] = None,
        return_adjs: bool = False,
        use_gpu: bool = False,
    ):
        """
        Compute the weighted k-nearest neighbors graph between the reference and query datasets

        Parameters
        ----------
        ref : np.ndarray
            The reference representation to build ref-query neighbor graph
        query : np.ndarray
            The query representation to build ref-query neighbor graph
        ref2 : np.ndarray
            The reference representation to build ref-ref neighbor graph
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
        return_adjs : bool
            Whether to return the adjacency matrices
        use_gpu : bool
        """

        wknn = get_wknn(
            ref=ref,
            query=query,
            ref2=ref2,
            k=k,
            query2ref=query2ref,
            ref2query=ref2query,
            weighting_scheme=weighting_scheme,
            top_n=top_n,
            return_adjs=return_adjs,
            use_gpu=use_gpu,
        )
