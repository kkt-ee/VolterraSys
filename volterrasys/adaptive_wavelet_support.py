"""Training-time support updates for adaptive sparse wavelet kernels."""

from math import ceil
from numbers import Real

import tensorflow as tf


class AdaptiveWaveletSupport:
    """Prune and regrow a fixed-size canonical wavelet-kernel support.

    The controller performs support updates explicitly between optimizer
    steps. It uses one batch to score active coefficients and inactive
    candidates. It never changes support from a layer's forward method.
    """

    def __init__(
        self,
        layer,
        replacement_fraction=0.1,
        candidate_multiplier=4,
        seed=None,
    ):
        if (
            isinstance(replacement_fraction, bool)
            or not isinstance(replacement_fraction, Real)
            or replacement_fraction <= 0
            or replacement_fraction > 1
        ):
            raise ValueError(
                "replacement_fraction must be in the interval (0, 1]."
            )
        if (
            isinstance(candidate_multiplier, bool)
            or not isinstance(candidate_multiplier, int)
            or candidate_multiplier < 1
        ):
            raise ValueError("candidate_multiplier must be a positive integer.")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int)
        ):
            raise ValueError("seed must be an integer or None.")

        self.layer = layer
        self.replacement_fraction = float(replacement_fraction)
        self.candidate_multiplier = candidate_multiplier
        self.seed = seed

    def refresh(
        self,
        model,
        inputs,
        targets,
        loss_fn,
        optimizer=None,
        sample_weight=None,
    ):
        """Replace weak active locations with gradient-scored candidates."""
        self.layer._require_adaptive_sparse_support()
        if not tf.executing_eagerly():
            raise RuntimeError("Adaptive support refresh must run eagerly.")
        if not callable(loss_fn):
            raise ValueError("loss_fn must be callable.")

        active_gradient, baseline_loss = self._kernel_gradient(
            model,
            inputs,
            targets,
            loss_fn,
            sample_weight,
        )
        values = tf.convert_to_tensor(self.layer.sparse_kernel_values)
        active_scores = tf.abs(values * active_gradient)
        tf.debugging.assert_all_finite(
            active_scores,
            "Active sparse-support scores contain nonfinite values.",
        )

        budget = self.layer.sparse_budget
        available = self.layer._adaptive_support_size - budget
        if available < 1:
            raise ValueError(
                "Adaptive support cannot change because every canonical "
                "coefficient is already active."
            )
        replacement_count = min(
            max(1, ceil(self.replacement_fraction * budget)),
            available,
        )
        candidate_count = min(
            budget,
            available,
            max(
                replacement_count,
                self.candidate_multiplier * replacement_count,
            ),
        )

        active_score_values = active_scores.numpy().tolist()
        weakest_positions = tuple(
            sorted(
                range(budget),
                key=lambda position: (
                    active_score_values[position],
                    position,
                ),
            )[:candidate_count]
        )
        update_number = int(
            self.layer.adaptive_sparse_update_count.numpy()
        )
        candidate_seed = (
            self.layer.sparse_seed
            if self.seed is None
            else self.seed
        ) + update_number + 1
        candidates = self.layer.sample_inactive_sparse_support(
            candidate_count,
            seed=candidate_seed,
        )

        original_support = self.layer.adaptive_sparse_support()
        saved_rows = tuple(
            original_support[position] for position in weakest_positions
        )
        saved_values = tf.gather(
            self.layer.sparse_kernel_values,
            weakest_positions,
        ).numpy()
        try:
            self.layer.replace_adaptive_sparse_support(
                weakest_positions,
                candidates,
            )
            candidate_gradient, _ = self._kernel_gradient(
                model,
                inputs,
                targets,
                loss_fn,
                sample_weight,
            )
            candidate_scores = tf.abs(
                tf.gather(candidate_gradient, weakest_positions)
            )
            tf.debugging.assert_all_finite(
                candidate_scores,
                "Candidate sparse-support scores contain nonfinite values.",
            )
            candidate_score_values = candidate_scores.numpy().tolist()
        finally:
            self.layer.replace_adaptive_sparse_support(
                weakest_positions,
                saved_rows,
                values=saved_values,
            )

        selected_candidates = tuple(
            sorted(
                range(candidate_count),
                key=lambda index: (
                    -candidate_score_values[index],
                    index,
                ),
            )[:replacement_count]
        )
        replaced_positions = weakest_positions[:replacement_count]
        grown_support = tuple(
            candidates[index] for index in selected_candidates
        )
        self.layer.replace_adaptive_sparse_support(
            replaced_positions,
            grown_support,
            increment_update_count=True,
        )
        reset_slot_count = self._reset_optimizer_state(
            optimizer,
            replaced_positions,
        )

        return {
            'baseline_loss': float(baseline_loss.numpy()),
            'replaced_positions': replaced_positions,
            'removed_support': tuple(
                original_support[position]
                for position in replaced_positions
            ),
            'grown_support': grown_support,
            'candidate_count': candidate_count,
            'optimizer_slots_reset': reset_slot_count,
        }

    def _kernel_gradient(
        self,
        model,
        inputs,
        targets,
        loss_fn,
        sample_weight,
    ):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=False)
            if sample_weight is None:
                loss = loss_fn(targets, predictions)
            else:
                loss = loss_fn(
                    targets,
                    predictions,
                    sample_weight=sample_weight,
                )
            loss = tf.reduce_mean(loss)
            if model.losses:
                loss = loss + tf.add_n(model.losses)
        gradient = tape.gradient(
            loss,
            self.layer.sparse_kernel_values,
        )
        if gradient is None:
            raise ValueError(
                "The loss has no gradient with respect to the adaptive "
                "sparse kernel values."
            )
        return gradient, loss

    def _reset_optimizer_state(self, optimizer, positions):
        if optimizer is None:
            return 0
        if not hasattr(optimizer, '_var_key'):
            raise ValueError(
                "The optimizer does not expose variable-specific state."
            )
        variable = self.layer.sparse_kernel_values
        variable_indices = getattr(
            optimizer,
            '_trainable_variables_indices',
            {},
        )
        variable_index = variable_indices.get(optimizer._var_key(variable))
        if variable_index is None:
            return 0

        trainable_variables = getattr(
            optimizer,
            '_trainable_variables',
            (),
        )
        indices = tf.constant(
            [[position] for position in positions],
            dtype=tf.int32,
        )
        reset_count = 0
        seen = set()
        for attribute, slots in vars(optimizer).items():
            if attribute == '_trainable_variables':
                continue
            if not isinstance(slots, (list, tuple)):
                continue
            if len(slots) != len(trainable_variables):
                continue
            slot = slots[variable_index]
            if not hasattr(slot, 'assign'):
                continue
            if tuple(slot.shape) != tuple(variable.shape):
                continue
            if id(slot) in seen:
                continue
            seen.add(id(slot))

            fill_value = 0.0
            if 'accumulator' in attribute:
                fill_value = float(
                    getattr(optimizer, 'initial_accumulator_value', 0.0)
                )
            updates = tf.fill(
                (len(positions),),
                tf.cast(fill_value, slot.dtype),
            )
            slot.assign(
                tf.tensor_scatter_nd_update(slot, indices, updates)
            )
            reset_count += 1
        return reset_count
