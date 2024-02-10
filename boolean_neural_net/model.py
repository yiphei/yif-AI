import copy
import random
from collections import defaultdict, deque
from itertools import chain, combinations

import torch


def generate_powerset_iterator(set_elements):
    return chain.from_iterable(
        combinations(set_elements, r) for r in range(len(set_elements), -1, -1)
    )


def combine_iterators(iterable_one, iterable_two):
    for item in iterable_one:
        if item:
            merged_set = set().union(*item)
            yield merged_set

    for item in iterable_two:
        yield item


class BooleanLayer:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        W_pos = torch.randint(
            0,
            2,
            (
                out_dim,
                in_dim,
            ),
        )
        W_neg = torch.randint(
            0,
            2,
            (
                out_dim,
                in_dim,
            ),
        )
        W_neg[W_pos == 1] = 0
        self.W = torch.cat((W_pos, W_neg), dim=1)

        # each W row must have at least one 1
        zero_row_idxs = (self.W.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        col_idxs = torch.randint(0, in_dim * 2, (zero_row_idxs.shape[0],))
        self.W[zero_row_idxs, col_idxs] = 1

        self.W_doubt = torch.zeros_like(self.W)

        self.out = None
        self.full_X = None

    def conjunction_mul(self, X, W):
        matrix_X = X.repeat(1, W.shape[0], 1)  # repeat X for each row of W
        mask = W == 1
        masked_X = torch.where(
            mask, matrix_X, torch.tensor(1)
        )  # theoretically, it should not be replaced with 1 (it should just be omitted), but mathematically it is fine because 1 is idempotent in multiplication
        return torch.prod(masked_X, dim=2, keepdim=True).view(X.shape[0], -1)

    def get_neg_col_idx(self, col_idx):
        return (col_idx + self.in_dim) % (self.in_dim * 2)

    def get_pos_col_idx(self, col_idx):
        return col_idx % self.in_dim

    def forward(self, X):
        X_neg = 1 - X
        self.full_X = torch.cat((X, X_neg), dim=1)
        self.out = self.conjunction_mul(self.full_X.unsqueeze(1), self.W)
        return self.out

    def _get_new_X_row_idxs_per_W_col(
        self,
        curr_depth,
        max_depth,
        curr_one_Y_row_state,
        prev_W_row_idx,
        q,
        W_row_to_one_Y_row_idxs,
    ):
        # the output is of shape [({1,2,3},{4,5,6}), ({2,3},{4,5,1,6}), ...] where ({1,2,3},{4,5,6}) means
        # that W[[1,2,3]][0] should be 1 and W[[4,5,6]][0] should be 0 and full_X[[1,2,3]][0] should be 1
        # and full_X[[4,5,6]][0] should be 0
        if curr_depth == max_depth or len(curr_one_Y_row_state) == 0:
            return [], len(curr_one_Y_row_state) == 0

        curr_W_row_idx = prev_W_row_idx
        while curr_W_row_idx not in curr_one_Y_row_state and q:
            curr_W_row_idx = q.popleft()

        curr_one_Y_idxs = W_row_to_one_Y_row_idxs[curr_W_row_idx]
        if not curr_one_Y_row_state[curr_W_row_idx]:
            # TODO: too ad-hoc, needs to be refactored
            updated_one_Y_row_state = copy.deepcopy(curr_one_Y_row_state)
            del updated_one_Y_row_state[curr_W_row_idx]
            sub_new_X_row_idxs_per_W_col, is_solved = (
                self._get_new_X_row_idxs_per_W_col(
                    curr_depth + 1,
                    max_depth,
                    updated_one_Y_row_state,
                    curr_W_row_idx,
                    copy.deepcopy(q),
                    W_row_to_one_Y_row_idxs,
                )
            )
            assert is_solved
            sub_new_X_row_idxs_per_W_col.append((curr_one_Y_idxs, set()))
            return sub_new_X_row_idxs_per_W_col, True

        remaining_q = list(q)
        remaining_one_Y_idxs = [
            W_row_to_one_Y_row_idxs[x]
            for x in remaining_q
            if len(W_row_to_one_Y_row_idxs[x] & curr_one_Y_idxs) == 0
            and len(W_row_to_one_Y_row_idxs[x] & curr_one_Y_row_state[curr_W_row_idx])
            > 0
        ]
        for opposite_set in combine_iterators(
            generate_powerset_iterator(remaining_one_Y_idxs),
            [curr_one_Y_row_state[curr_W_row_idx]],
        ):
            remaining_Y_idxs = set(range(self.full_X.shape[0])) - (
                opposite_set | curr_one_Y_idxs
            )
            sub_remaining_one_Y_idxs = [
                x for x in remaining_one_Y_idxs if x.issubset(remaining_Y_idxs)
            ]

            for remaining_merged_set in combine_iterators(
                generate_powerset_iterator(sub_remaining_one_Y_idxs), [set()]
            ):
                complement_remaining_Y_subset = remaining_Y_idxs - remaining_merged_set

                first_left_W = curr_one_Y_idxs | complement_remaining_Y_subset
                first_right_W = opposite_set | remaining_merged_set

                second_left_W = curr_one_Y_idxs | remaining_merged_set
                second_right_W = opposite_set | complement_remaining_Y_subset

                for left_W, right_W in [
                    (first_left_W, first_right_W),
                    (second_left_W, second_right_W),
                ]:
                    updated_one_Y_row_state = {}
                    for k, v in curr_one_Y_row_state.items():
                        one_Y_idxs = W_row_to_one_Y_row_idxs[k]
                        sub_diff = v

                        if one_Y_idxs.issubset(left_W):
                            sub_diff = v - right_W
                        elif one_Y_idxs.issubset(right_W):
                            sub_diff = v - left_W

                        # implicit here is the removal of one_Y_idxs for which there is no unresolved zero Y row idxs left
                        if len(sub_diff) > 0:
                            updated_one_Y_row_state[k] = sub_diff

                    sub_new_X_row_idxs_per_W_col, is_solved = (
                        self._get_new_X_row_idxs_per_W_col(
                            curr_depth + 1,
                            max_depth,
                            updated_one_Y_row_state,
                            curr_W_row_idx,
                            copy.deepcopy(q),
                            W_row_to_one_Y_row_idxs,
                        )
                    )
                    if is_solved:
                        sub_new_X_row_idxs_per_W_col.append((left_W, right_W))
                        return sub_new_X_row_idxs_per_W_col, True

        return [], False

    def _get_W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum(
        self,
        W_row_idxs_set_idxs,
        max_sorted_idx_per_W_row_idxs_set_idxs,
        used_W_col_idxs,
        max_sum,
        sorted_W_row_idxs_sets_doubt_sum_per_col_indices,
        offset_sorted_W_row_idxs_sets_doubt_sum_per_col,
        new_X_row_idxs_per_W_col,
        W_row_idxs_set_sequencing,
    ):
        # This is the core function that determines the best W column assignment based on W_doubt. Before was all preprocessing for a faster algorithm.
        # The output shape is {0:1, 1:0, 2:5} where 0:1 means that one_W_row_idxs and zero_W_row_idxs pair indexed at 0 should be assigned to W column 1

        if len(W_row_idxs_set_idxs) == 1:
            W_row_idxs_set_idx = list(W_row_idxs_set_idxs)[0]
            max_sorted_idx = (
                max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]
                if max_sorted_idx_per_W_row_idxs_set_idxs is not None
                else self.in_dim - 1
            )

            for sorted_idx in range(max_sorted_idx + 1):
                col_idx = sorted_W_row_idxs_sets_doubt_sum_per_col_indices[
                    W_row_idxs_set_idx, sorted_idx
                ].item()
                W_row_idxs_doubt_sum = offset_sorted_W_row_idxs_sets_doubt_sum_per_col[
                    W_row_idxs_set_idx, sorted_idx
                ].item()

                if (
                    max_sum is not None and W_row_idxs_doubt_sum >= max_sum
                ):  # theoretically, this would never evaluate to True
                    return None, None
                if self.get_pos_col_idx(col_idx) not in used_W_col_idxs:
                    return W_row_idxs_doubt_sum, {W_row_idxs_set_idx: sorted_idx}

            return None, None

        curr_max_sorted_idx_per_W_row_idxs_set_idxs = [-1] * len(
            new_X_row_idxs_per_W_col
        )
        min_doubt_sum = max_sum
        W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum = None

        for i in range(len(W_row_idxs_set_sequencing)):
            W_row_idxs_set_idx = W_row_idxs_set_sequencing[i]
            if W_row_idxs_set_idx not in W_row_idxs_set_idxs:
                continue

            curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx] += 1
            if (
                max_sorted_idx_per_W_row_idxs_set_idxs is not None
                and curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]
                > max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]
            ):
                return (
                    min_doubt_sum,
                    W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum,
                )

            col_idx = sorted_W_row_idxs_sets_doubt_sum_per_col_indices[
                W_row_idxs_set_idx,
                curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx],
            ].item()
            W_row_idxs_doubt_sum = offset_sorted_W_row_idxs_sets_doubt_sum_per_col[
                W_row_idxs_set_idx,
                curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx],
            ].item()
            if min_doubt_sum is not None and W_row_idxs_doubt_sum >= min_doubt_sum:
                return (
                    min_doubt_sum,
                    W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum,
                )

            if self.get_pos_col_idx(col_idx) not in used_W_col_idxs:
                updated_used_col_idxs = used_W_col_idxs | {
                    self.get_pos_col_idx(col_idx)
                }
                new_max_sum = (
                    max_sum - W_row_idxs_doubt_sum if max_sum is not None else None
                )
                (
                    sub_min_doubt_sum,
                    sub_W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum,
                ) = self._get_W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum(
                    W_row_idxs_set_idxs - {W_row_idxs_set_idx},
                    curr_max_sorted_idx_per_W_row_idxs_set_idxs,
                    updated_used_col_idxs,
                    new_max_sum,
                    sorted_W_row_idxs_sets_doubt_sum_per_col_indices,
                    offset_sorted_W_row_idxs_sets_doubt_sum_per_col,
                    new_X_row_idxs_per_W_col,
                    W_row_idxs_set_sequencing,
                )

                if sub_min_doubt_sum is not None and (
                    min_doubt_sum is None
                    or sub_min_doubt_sum + W_row_idxs_doubt_sum < min_doubt_sum
                ):
                    min_doubt_sum = W_row_idxs_doubt_sum + sub_min_doubt_sum
                    W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum = (
                        sub_W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum
                    )
                    W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum[
                        W_row_idxs_set_idx
                    ] = curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]

        return min_doubt_sum, W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum

    def backprop(self, Y, is_first_layer=False):
        if torch.equal(Y, self.out):
            return None

        self.W_doubt[
            self.W == 1
        ] += 1  # penalize existing Ws that are 1 by increasing doubt

        W_row_to_zero_Y_row_idxs = {}
        W_row_to_one_Y_row_idxs = {}
        for i in range(self.W.shape[0]):
            row_Y = Y[:, i]

            zero_Y_idxs = torch.nonzero(row_Y == 0).squeeze(1).tolist()
            if zero_Y_idxs:
                W_row_to_zero_Y_row_idxs[i] = set(zero_Y_idxs)

            one_Y_idxs = torch.nonzero(row_Y == 1).squeeze(1).tolist()
            if one_Y_idxs:
                W_row_to_one_Y_row_idxs[i] = set(one_Y_idxs)

        W_rows_of_unique_one_Y_row_idxs = (
            set()
        )  # this is constructed by assigning one_Y_row_idxs to W columns incrementally (from 0). This will later be changed
        visited_one_Y_row_idxs = set()

        W_col_to_new_X_row_idxs = {}
        for W_row_idx, one_Y_row_idxs in W_row_to_one_Y_row_idxs.items():
            tuple_value = tuple(one_Y_row_idxs)
            if tuple_value not in visited_one_Y_row_idxs:
                visited_one_Y_row_idxs.add(tuple_value)
                W_rows_of_unique_one_Y_row_idxs.add(W_row_idx)

        if is_first_layer:
            # In the first layer, full_X corresponds to the input of the model, so you can't alter full_X
            W_col_to_new_X_row_idxs = {}
            for col_idx in range(self.in_dim):
                one_idxs = set(
                    (self.full_X[:, col_idx] == 1).nonzero().squeeze(1).tolist()
                )
                zero_idxs = set(
                    (self.full_X[:, col_idx] == 0).nonzero().squeeze(1).tolist()
                )
                W_col_to_new_X_row_idxs[col_idx] = (one_idxs, zero_idxs)

        elif W_rows_of_unique_one_Y_row_idxs:
            one_Y_row_state = {
                W_row: W_row_to_zero_Y_row_idxs.get(W_row, set())
                for W_row in W_rows_of_unique_one_Y_row_idxs
            }  # this tracks unresolved zero Y row idxs for each W row idx
            sorted_one_Y_row_idxs = sorted(
                list(W_rows_of_unique_one_Y_row_idxs),
                key=lambda x: len(W_row_to_one_Y_row_idxs[x]),
                reverse=True,
            )  # a heuristical optimization to address the largest one_Y_row_idxs first
            q = deque(sorted_one_Y_row_idxs)

            new_X_row_idxs_per_W_col, is_solved = self._get_new_X_row_idxs_per_W_col(
                0, self.in_dim, one_Y_row_state, q.popleft(), q, W_row_to_one_Y_row_idxs
            )  # X_row_idxs_per_W_col does not necessarily contain a slot for each col
            assert is_solved

            # new_X_row_idxs_per_W_col provides a valid update of W and full_X that satisfies the expected Y.
            # However, we assigned one_Y_row_idxs to W columns incrementally (from 0) for simplicity.
            # Below, we determine the best W column assignment based on W_doubt.

            W_row_idxs_per_col = defaultdict(
                lambda: [[], []]
            )  # this represents all one_W_row_idxs and zero_W_row_idxs pairs
            for W_row_idx, one_Y_row_idxs in W_row_to_one_Y_row_idxs.items():
                for W_col_idx, new_X_row_idxs in enumerate(new_X_row_idxs_per_W_col):
                    if one_Y_row_idxs.issubset(new_X_row_idxs[0]):
                        W_row_idxs_per_col[W_col_idx][0].append(W_row_idx)
                    elif one_Y_row_idxs.issubset(new_X_row_idxs[1]):
                        W_row_idxs_per_col[W_col_idx][1].append(W_row_idx)

            # calculate the W_doubt sum of each one_W_row_idxs and zero_W_row_idxs pairs for all W columns
            W_row_idxs_sets_doubt_sum_per_col = []
            for W_row_idxs in W_row_idxs_per_col.keys():
                sums = self.W_doubt[W_row_idxs_per_col[W_row_idxs][0]].sum(dim=0)
                neg_sum = torch.roll(
                    self.W_doubt[W_row_idxs_per_col[W_row_idxs][1]].sum(dim=0),
                    shifts=-self.in_dim,
                    dims=0,
                )
                sums += neg_sum
                W_row_idxs_sets_doubt_sum_per_col.append(sums)

            W_row_idxs_sets_doubt_sum_per_col = torch.stack(
                W_row_idxs_sets_doubt_sum_per_col
            )
            sorted_W_row_idxs_sets_doubt_sum_per_col = torch.sort(
                W_row_idxs_sets_doubt_sum_per_col, dim=1, descending=False
            )  # sort by increasing sum

            # Prune W columns
            opt_values = []
            opt_indices = []

            for sum_values, sum_indices in zip(
                sorted_W_row_idxs_sets_doubt_sum_per_col.values,
                sorted_W_row_idxs_sets_doubt_sum_per_col.indices,
            ):
                opt_sums = []
                opt_sum_idxs = []
                visited_col_idxs = set()
                for sum_value, idx in zip(sum_values, sum_indices):
                    idx_value = idx.item()
                    if self.get_pos_col_idx(idx_value) not in visited_col_idxs:
                        opt_sums.append(sum_value.item())
                        opt_sum_idxs.append(idx_value)
                        visited_col_idxs.add(self.get_pos_col_idx(idx_value))

                    if len(visited_col_idxs) == self.in_dim:
                        break

                opt_values.append(opt_sums)
                opt_indices.append(opt_sum_idxs)

            sorted_W_row_idxs_sets_doubt_sum_per_col_values = torch.tensor(opt_values)
            sorted_W_row_idxs_sets_doubt_sum_per_col_indices = torch.tensor(opt_indices)

            # a heuristical optimization that sorts W columns by increasing offset sum across one_W_row_idxs and zero_W_row_idxs pairs
            offset_sorted_W_row_idxs_sets_doubt_sum_per_col = (
                sorted_W_row_idxs_sets_doubt_sum_per_col_values
                - sorted_W_row_idxs_sets_doubt_sum_per_col_values[:, 0].unsqueeze(1)
            )  # normalize the sum by subtracting the smallest sum
            offset_W_row_idxs_sets_doubt_sum_to_cols_dict = defaultdict(list)
            for col_idx, offset_sums in enumerate(
                offset_sorted_W_row_idxs_sets_doubt_sum_per_col
            ):
                for offset_sum in offset_sums:
                    offset_W_row_idxs_sets_doubt_sum_to_cols_dict[
                        offset_sum.item()
                    ].append(col_idx)
            sorted_W_row_idxs_sets_doubt_sum = sorted(
                offset_W_row_idxs_sets_doubt_sum_to_cols_dict.keys()
            )
            W_row_idxs_set_sequencing = [
                offset_W_row_idxs_sets_doubt_sum_to_cols_dict[x]
                for x in sorted_W_row_idxs_sets_doubt_sum
            ]  # based on increasing offset W row idxs sets sum
            W_row_idxs_set_sequencing = [
                x for sublist in W_row_idxs_set_sequencing for x in sublist
            ]  # flatten

            _, W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum = (
                self._get_W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum(
                    set(range(len(new_X_row_idxs_per_W_col))),
                    None,
                    set(),
                    None,
                    sorted_W_row_idxs_sets_doubt_sum_per_col_indices,
                    offset_sorted_W_row_idxs_sets_doubt_sum_per_col,
                    new_X_row_idxs_per_W_col,
                    W_row_idxs_set_sequencing,
                )
            )
            assert W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum is not None

            # given the optimal assignment, we recreate new_X_row_idxs_per_W_col in W_col_to_new_X_row_idxs
            for (
                W_row_idxs_set_idx,
                sorted_idx,
            ) in W_row_idxs_set_idx_to_sorted_col_idx_w_min_doubt_sum.items():
                old_col_idx = W_row_idxs_set_idx
                new_col_idx = sorted_W_row_idxs_sets_doubt_sum_per_col_indices[
                    W_row_idxs_set_idx, sorted_idx
                ].item()
                new_pos_col_idx = self.get_pos_col_idx(new_col_idx)

                new_X_row_idxs = new_X_row_idxs_per_W_col[old_col_idx]
                if new_pos_col_idx != new_col_idx:
                    new_X_row_idxs = (new_X_row_idxs[1], new_X_row_idxs[0])
                W_col_to_new_X_row_idxs[new_pos_col_idx] = new_X_row_idxs

        # Y columns with zero 1s need to be treated differently
        W_col_to_new_X_row_idxs_for_zero_Y = {}
        W_row_idxs_with_zero_Ys = list(
            set(range(self.W.shape[0])) - (W_row_to_one_Y_row_idxs.keys())
        )
        if W_row_idxs_with_zero_Ys:
            # TODO: too ad-hoc, needs to be refactored
            available_cols = self.in_dim - len(W_col_to_new_X_row_idxs.keys())
            if available_cols > 0:
                # TODO: this is a problem if you have identical rows of full_X
                new_X_row_idxs = list(range(self.full_X.shape[0]))
                partitions = random.randint(1, min(available_cols, len(new_X_row_idxs)))
                selected_partition_idxs = [0]
                if partitions > 1:
                    selected_partition_idxs = sorted(
                        random.sample(set(new_X_row_idxs) - {0}, partitions - 1)
                    )

                last_partition_idx = 0
                X_row_partitions = []
                for partition_idx in selected_partition_idxs + [len(new_X_row_idxs)]:
                    row_partition = set(
                        new_X_row_idxs[last_partition_idx:partition_idx]
                    )
                    if row_partition:
                        complement_partition = set(new_X_row_idxs) - row_partition
                        X_row_partitions.append((complement_partition, row_partition))

                    last_partition_idx = partition_idx

                used_col_idxs = W_col_to_new_X_row_idxs.keys()
                pos_used_col_idxs = set(
                    [self.get_pos_col_idx(col_idx) for col_idx in used_col_idxs]
                )
                available_col_idxs = set(range(self.in_dim)) - pos_used_col_idxs
                sums = torch.sort(
                    self.W_doubt[W_row_idxs_with_zero_Ys].sum(dim=0),
                    dim=0,
                    descending=False,
                )

                for col_idx_tensor in sums.indices:
                    col_idx = col_idx_tensor.item()
                    neg_col_idx = self.get_neg_col_idx(col_idx)
                    if (
                        self.get_pos_col_idx(col_idx) in available_col_idxs
                        and neg_col_idx not in W_col_to_new_X_row_idxs_for_zero_Y
                    ):
                        W_col_to_new_X_row_idxs_for_zero_Y[col_idx] = X_row_partitions[
                            len(W_col_to_new_X_row_idxs_for_zero_Y.keys())
                        ]
                        if len(W_col_to_new_X_row_idxs_for_zero_Y.keys()) == partitions:
                            break
            else:

                def find_best_1s_col_idxs(depth, max_depth, curre_sol, remaining_rows):
                    if depth == max_depth or len(remaining_rows) == 0:
                        return curre_sol if len(remaining_rows) == 0 else None

                    for W_col_idx, X_row_idxs in W_col_to_new_X_row_idxs.items():
                        neg_W_col_idx = self.get_neg_col_idx(W_col_idx)
                        sub = remaining_rows - X_row_idxs[1]
                        if (
                            W_col_idx not in curre_sol
                            and neg_W_col_idx not in curre_sol
                            and len(sub) < len(remaining_rows)
                        ):
                            sol = find_best_1s_col_idxs(
                                depth + 1, max_depth, curre_sol | {W_col_idx}, sub
                            )
                            if sol is not None:
                                return sol

                        sub = remaining_rows - X_row_idxs[0]
                        if (
                            W_col_idx not in curre_sol
                            and neg_W_col_idx not in curre_sol
                            and len(sub) < len(remaining_rows)
                        ):
                            sol = find_best_1s_col_idxs(
                                depth + 1, max_depth, curre_sol | {neg_W_col_idx}, sub
                            )
                            if sol is not None:
                                return sol
                    return None

                W_col_idxs = find_best_1s_col_idxs(
                    0, self.in_dim, set(), set(range(self.full_X.shape[0]))
                )
                assert W_col_idxs is not None
                for W_col_idx in W_col_idxs:
                    pos_col_idx = self.get_pos_col_idx(W_col_idx)
                    target_X_values = W_col_to_new_X_row_idxs[pos_col_idx]
                    if W_col_idx != pos_col_idx:
                        target_X_values = (target_X_values[1], target_X_values[0])
                    W_col_to_new_X_row_idxs_for_zero_Y[W_col_idx] = target_X_values

        updated_W = torch.zeros_like(self.W)
        for W_row_idx, Y_row_idxs in W_row_to_one_Y_row_idxs.items():
            for W_col_idx, new_X_row_idxs in W_col_to_new_X_row_idxs.items():
                if Y_row_idxs.issubset(new_X_row_idxs[0]):
                    updated_W[W_row_idx, W_col_idx] = 1
                elif Y_row_idxs.issubset(new_X_row_idxs[1]):
                    updated_W[W_row_idx, W_col_idx + self.in_dim] = 1

        for W_row_idx in W_row_idxs_with_zero_Ys:
            updated_W[W_row_idx, list(W_col_to_new_X_row_idxs_for_zero_Y.keys())] = 1

        self.W = updated_W

        updated_full_X = None
        if not is_first_layer:
            updated_full_X = torch.zeros_like(self.full_X)
            for W_col_idx, new_X_row_idxs in W_col_to_new_X_row_idxs.items():
                updated_full_X[list(new_X_row_idxs[0]), W_col_idx] = 1

            for W_col_idx, new_X_row_idxs in W_col_to_new_X_row_idxs_for_zero_Y.items():
                target_X_row_idxs = new_X_row_idxs[0]
                target_complement_X_row_idxs = new_X_row_idxs[1]
                target_W_col_idx = self.get_pos_col_idx(W_col_idx)
                if W_col_idx >= self.in_dim:
                    target_X_row_idxs = new_X_row_idxs[1]
                    target_complement_X_row_idxs = new_X_row_idxs[0]

                updated_full_X[list(target_X_row_idxs), target_W_col_idx] = 1
                updated_full_X[list(target_complement_X_row_idxs), target_W_col_idx] = 0

        return updated_full_X[:, : self.in_dim] if not is_first_layer else None


class BooleanNN:
    def __init__(self, in_dim, clause_dim):
        hidden_dim = 1
        self.input_layer = BooleanLayer(in_dim, clause_dim)
        self.hidden_layers = [
            BooleanLayer(clause_dim, clause_dim) for _ in range(hidden_dim)
        ]
        self.output_layer = BooleanLayer(clause_dim, 1)
        self.out = None

    def forward(self, X):
        X = self.input_layer.forward(X)
        for layer in self.hidden_layers:
            X = layer.forward(X)
        X = self.output_layer.forward(X)
        self.out = X.squeeze(1)
        return self.out

    def backprop(self, y):
        y = y.unsqueeze(1)

        updated_X = self.output_layer.backprop(y)
        if updated_X is None:
            return

        for layer in reversed(self.hidden_layers):
            updated_X = layer.backprop(updated_X)
            if updated_X is None:
                return

        self.input_layer.backprop(updated_X, True)
