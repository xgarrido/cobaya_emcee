"""
.. module:: samplers.emcee

:Synopsis: Goodman & Weare's Affine Invariant Markov chain Monte Carlo Ensemble sampler
:Author: Daniel Foreman-Mackey, David W. Hogg, Dustin Lang, Jonathan Goodman (wrapped for cobaya by Xavier Garrido)
"""
import os
import re
import sys
from typing import Union

import numpy as np
from cobaya.collection import SampleCollection
from cobaya.conventions import Extension
from cobaya.log import LoggedError, is_debug
from cobaya.sampler import Sampler
from cobaya.tools import NumberWithUnits, load_module

Extension.hdf5 = ".h5"


class EMCEE(Sampler):
    r"Goodman & Weare's Affine Invariant Markov chain Monte Carlo Ensemble sample \cite{emcee2013}"

    file_base_name = "emcee"

    # instance variables from yaml
    nwalkers: Union[int, NumberWithUnits]
    nsteps: Union[int, NumberWithUnits]
    thin_by: int
    moves: dict
    progress: bool
    output_format: str

    def set_instance_defaults(self):
        super().set_instance_defaults()
        self.parallel = dict(kind="")
        self.ndim = 0
        self.nblobs = 0

    def initialize(self):
        self.ndim = self.model.prior.d()
        if self.ndim == 0:
            raise LoggedError(self.log, "No parameters being varied for sampler")
        self.log.debug("Number of parameters to sample: %i", self.ndim)

        for p in ["nwalkers", "nsteps"]:
            if not isinstance(getattr(self, p), int):
                setattr(
                    self,
                    p,
                    NumberWithUnits(getattr(self, p), "d", dtype=int, scale=self.ndim).value,
                )
            self.log.debug("Number of %s: %i", p, getattr(self, p))

        # Load emcee module
        self.emcee = load_module("emcee")

        # Import additional modules for parallel computing if requested
        pool = None
        if not is_debug():
            # Multithreading parallel computing via python-native multiprocessing module
            if self.parallel.get("kind").lower() == "multiprocessing":
                from multiprocessing import Pool

                pool = Pool(
                    self.parallel.get("args", dict(threads=1)).get("threads")
                )  # number of threads chosen by user
            # MPI parallel computing via external schwimmbad module
            elif self.parallel.get("kind").lower() == "mpi":
                from schwimmbad import MPIPool

                pool = MPIPool()
                if not pool.is_master():  # Necessary bit for MPI
                    pool.wait()
                    sys.exit(0)

        # Prepare some inputs for the MCMC
        blobs_dtype = [("logprior", float)]
        blobs_dtype += [(f"loglike_{name}", float) for name in self.model.likelihood]
        blobs_dtype += [
            (f"{derived}", float) for derived in self.model.parameterization.derived_params().keys()
        ]
        self.nblobs = len(blobs_dtype)

        # Initialize output file
        backend = None
        if self.output:
            if self.output_format == "hdf5":
                backend = self.emcee.backends.HDFBackend(
                    os.path.join(self.output.folder, self.output.prefix + Extension.hdf5)
                )
                if not self.output.is_resuming():
                    backend.reset(self.nwalkers, self.ndim)
                else:
                    self.log.info("Loading %i previous steps...", backend.iteration)
            elif self.output_format == "txt":
                # One collection per walker
                self.collection = [
                    SampleCollection(
                        self.model, self.output, name=str(i), resuming=self.output.is_resuming()
                    )
                    for i in range(1, self.nwalkers + 1)
                ]
            else:
                raise LoggedError(
                    self.log,
                    f"Unkown '{self.output_format}' output format! "
                    "Must be either 'hdf5' or 'txt'.",
                )

        # Initialize the move schedule
        moves = getattr(self.emcee.moves, self.moves.get("kind", "StretchMove"))
        moves = moves(**self.moves.get("args", {}))
        self.sampler = self.emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self._wrapped_logposterior,
            moves=moves,
            pool=pool,
            backend=backend,
            blobs_dtype=blobs_dtype,
        )
        self.mpi_info("Initialized!")

    def run(self):

        kwargs = dict(
            initial_state=self.get_initial_state(),
            iterations=self.nsteps,
            thin_by=self.thin_by,
            progress=self.progress,
        )

        for istep, result in enumerate(self.sampler.sample(**kwargs)):
            if istep == 0:
                nfinite = np.isfinite(result.log_prob).sum()
                if nfinite < 2:
                    raise LoggedError(
                        self.log,
                        "Your chain cannot progress: "
                        "less than 2 of your walkers are starting at a finite value of the posterior. "
                        "Please check if your starting positions are correct, and/or use "
                        "debug mode to check your likelihoods.",
                    )
                if nfinite < (0.5 * self.nwalkers):
                    self.log.warning(
                        "Warning, your chain will take time to converge: "
                        "only %i%% of your walkers are starting "
                        "at a finite value of the posterior. "
                        "Please check if your starting positions are correct, and/or use "
                        "debug mode to check your likelihoods.",
                        nfinite * 100 / self.nwalkers,
                    )

            if self.output_format == "txt":
                blobs_arr = result.blobs.view(dtype=np.float64).reshape(self.nwalkers, -1)
                nlkl = len(self.model.likelihood)
                for iwalk in range(self.nwalkers):
                    blobs = blobs_arr[iwalk]
                    self.collection[iwalk].add(
                        values=result.coords[iwalk],
                        logpost=result.log_prob[iwalk],
                        logpriors=blobs[:1],
                        loglikes=blobs[1 : 1 + nlkl],
                        derived=blobs[1 + nlkl :],
                    )
                    self.collection[iwalk].out_update()

        self.mpi_info("Reached maximum number of steps allowed (%s). Stopping.", self.nsteps)

    def get_initial_state(self):
        """
        Get initial/starting points either from paramters PDF or by loading the initial state from
        previous runs
        """
        if self.output_format == "hdf5" and self.output.is_resuming():
            return self.sampler._previous_state

        initial_state = np.empty((self.nwalkers, self.ndim))
        for i in range(self.nwalkers):
            if self.output_format == "txt" and self.output.is_resuming():
                last = len(self.collection[i]) - 1
                initial_point = (
                    self.collection[i][self.collection[i].sampled_params].iloc[last]
                ).to_numpy(dtype=np.float64, copy=True)
            else:
                initial_point, _results = self.model.get_valid_point(
                    max_tries=100 * self.ndim, random_state=self._rng
                )
                # initial_point = self.model.prior.reference()
            initial_state[i] = initial_point

        return initial_state

    def _wrapped_logposterior(self, param_values):
        results = self.model.logposterior(param_values)
        if results.logpost == -np.inf:
            results = [-np.inf] * (1 + self.nblobs)
        else:
            results = (
                [results.logpost] + results.logpriors + results.loglikes.tolist() + results.derived
            )
        return tuple(results)

    # Class methods
    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        regexps = [output.collection_regexp(name=None)]
        # if minimal:
        #     return [(r, None) for r in regexps]
        regexps += [
            re.compile(output.prefix_regexp_str + re.escape(ext.lstrip(".")) + "$")
            for ext in [Extension.hdf5]
        ]
        return [(r, None) for r in regexps]
