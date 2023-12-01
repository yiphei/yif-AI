import torch
import random
import math
import copy

from itertools import combinations, chain
from collections import deque

def generate_subsets(set_elements, subset_size):
    return [set(x) for x in list(combinations(set_elements, subset_size))]

def generate_powerset(set_elements):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(set_elements)
    return [set(x) for x in list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))]

class TsetlinBase:
    def conjunction_mul(self, X, W):
        matrix_X = X.repeat(1, W.shape[0], 1)
        mask = W > 0 # TODO: Right now this is not a problem because whenever I make update W or X, the negation is always zero. But if I change that, I will prob need to compare and choose the clause with the highest weight
        masked_X = torch.where(mask, matrix_X, torch.tensor(1)) # theoretically, you should not replace it with 1 (it should just be omitted), but mathematically it works out fine because an extra 1 does not change the output of the multiplication
        return torch.prod(masked_X, dim=2, keepdim=True).view(X.shape[0],-1)

class TsetlinLayer(TsetlinBase):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        W_pos = torch.randint(0, 2, (out_dim, in_dim,))
        W_neg = torch.randint(0, 2, (out_dim, in_dim,))
        W_neg[W_pos == 1] = 0
        self.W = torch.cat((W_pos, W_neg), dim=1)
        self.W_confidence = torch.zeros_like(self.W)
        zero_row_idxs = (self.W.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        col_idxs = torch.randint(0, in_dim * 2, (zero_row_idxs.shape[0],))
        self.W[zero_row_idxs, col_idxs] = 1

        self.out = None
        self.full_X = None

    def forward(self, X):
        X_neg = 1 - X
        self.full_X = torch.cat((X, X_neg), dim=1)
        self.out = self.conjunction_mul(self.full_X.unsqueeze(1), self.W)
        return self.out

    def update(self, Y, is_first_layer = False):
        if torch.equal(Y, self.out):
            return self.full_X[:,:self.in_dim]
        
        self.W_confidence[self.W > 0] += 1

        zero_Y_row_idxs_per_W_row = []
        one_Y_row_idxs_per_W_row = []
        for i in range(self.W.shape[0]):
            row_Y = Y[:, i]
            
            zero_Y_idxs = torch.nonzero(row_Y == 0).squeeze(1).tolist()
            zero_Y_row_idxs_per_W_row.append(set(zero_Y_idxs))

            one_Y_idxs = torch.nonzero(row_Y == 1).squeeze(1).tolist()
            one_Y_row_idxs_per_W_row.append(set(one_Y_idxs))

        return self.update_batch_non_first_layer(one_Y_row_idxs_per_W_row, zero_Y_row_idxs_per_W_row, is_first_layer)

    def update_batch_non_first_layer(self, one_Y_row_idxs_per_W_row, zero_Y_row_idxs_per_W_row, is_first_layer):
        unique_one_Y_row_idxs = set()
        visited_ones = set()

        W_col_to_new_X_row_idxs = {}
        for i,x in enumerate(one_Y_row_idxs_per_W_row):
            if x:
                tuple_x = tuple(x)
                if tuple_x not in visited_ones:
                    visited_ones.add(tuple_x)
                    unique_one_Y_row_idxs.add(i)

        if is_first_layer:
            W_col_to_new_X_row_idxs = {}
            for col_idx in range(self.full_X.shape[1]//2):
                one_idxs = set((self.full_X[:, col_idx] == 1).nonzero().squeeze(1).tolist())
                zero_idxs = set((self.full_X[:, col_idx] == 0).nonzero().squeeze(1).tolist())
                W_col_to_new_X_row_idxs[col_idx] = ((one_idxs, zero_idxs))

        elif unique_one_Y_row_idxs:
            tracking = {x: zero_Y_row_idxs_per_W_row[x] for x in unique_one_Y_row_idxs}
            sorted_one_Y_row_idxs = sorted(list(unique_one_Y_row_idxs), key=lambda x: len(one_Y_row_idxs_per_W_row[x]), reverse=True)
            q = deque(sorted_one_Y_row_idxs)

            def recursive_helper(depth, max_depth, current_solution, prev_W_row_idx, q):
                if depth == max_depth or len(current_solution) == 0:
                    return [], len(current_solution) == 0

                curr_W_row_idx = prev_W_row_idx
                while curr_W_row_idx not in current_solution and q:
                    curr_W_row_idx = q.popleft()

                curr_one_Y_idxs = one_Y_row_idxs_per_W_row[curr_W_row_idx]
                min_zero_Y_idxs_len = math.ceil(len(current_solution[curr_W_row_idx]) / (max_depth - depth))
                min_zero_Y_subsets = generate_subsets(current_solution[curr_W_row_idx], min(min_zero_Y_idxs_len, len(current_solution[curr_W_row_idx])))

                ordered_min_zero_Y_subsets = []
                remaining_q = list(q)
                for idx in remaining_q:
                    one_Y_idx = one_Y_row_idxs_per_W_row[idx]
                    if len(one_Y_idx) == min_zero_Y_idxs_len and len(one_Y_idx & curr_one_Y_idxs) == 0 and len(one_Y_idx & current_solution[curr_W_row_idx]) > 0:
                        ordered_min_zero_Y_subsets.append(one_Y_idx)

                for subset in min_zero_Y_subsets:
                    if subset not in ordered_min_zero_Y_subsets:
                        ordered_min_zero_Y_subsets.append(subset)

                for min_zero_Y_subset in ordered_min_zero_Y_subsets:
                    remaining_Y_idxs = set(range(self.full_X.shape[0])) - (min_zero_Y_subset | curr_one_Y_idxs)
                    remaining_Y_subsets = generate_powerset(remaining_Y_idxs)
                    remaining_Y_subsets.sort(key=lambda x: len(x), reverse=True)

                    remaining_Y_subsets_ordered = []
                    for idx in remaining_q:
                        one_Y_idx = one_Y_row_idxs_per_W_row[idx]
                        if one_Y_idx.issubset(remaining_Y_idxs):
                            remaining_Y_subsets_ordered.append(one_Y_idx)

                    for subset in remaining_Y_subsets:
                        if subset not in remaining_Y_subsets_ordered:
                            remaining_Y_subsets_ordered.append(subset)

                    for remaining_Y_subset in remaining_Y_subsets_ordered:
                        opposite_remaining_Y_subset = remaining_Y_idxs - remaining_Y_subset

                        #add remaining with the opposite
                        first_left_W = curr_one_Y_idxs | opposite_remaining_Y_subset
                        first_right_W = min_zero_Y_subset | remaining_Y_subset

                        second_left_W = curr_one_Y_idxs | remaining_Y_subset
                        second_right_W = min_zero_Y_subset | opposite_remaining_Y_subset

                        for left_W, right_W in [(first_left_W, first_right_W), (second_left_W, second_right_W)]:
                            updated_solution = {}
                            for k,v in current_solution.items():
                                one_Y_idxs = one_Y_row_idxs_per_W_row[k]
                                if one_Y_idxs.issubset(left_W):
                                    sub = v - right_W
                                    if len(sub) > 0:
                                        updated_solution[k] = sub
                                elif one_Y_idxs.issubset(right_W):
                                    sub = v - left_W
                                    if len(sub) > 0:
                                        updated_solution[k] = sub
                                else:
                                    updated_solution[k] = v

                            next_cols, solved = recursive_helper(depth+1, max_depth, updated_solution, curr_W_row_idx, copy.deepcopy(q))
                            if solved:
                                combined_cols = next_cols
                                combined_cols.append((left_W, right_W))
                                return combined_cols, True
                            
                return [], False
            
            X_row_idxs_per_W_col, solved = recursive_helper(0, self.in_dim, tracking, q.popleft(), q) # X_row_idxs_per_W_col does not necessarily contain a slot for each col
            assert solved

            # START - finding best col config based on W_confidence
            W_row_idxs_per_col = [ [[] for _ in range(2)] for _ in range(len(X_row_idxs_per_W_col))]

            for W_row_idx, one_Y_row_idxs in enumerate(one_Y_row_idxs_per_W_row):
                if one_Y_row_idxs:
                    for W_col_idx, X_row_idxs in enumerate(X_row_idxs_per_W_col):
                        if one_Y_row_idxs.issubset(X_row_idxs[0]):
                            W_row_idxs_per_col[W_col_idx][0].append(W_row_idx)
                        elif one_Y_row_idxs.issubset(X_row_idxs[1]):
                            W_row_idxs_per_col[W_col_idx][1].append(W_row_idx)

            W_row_idxs_sets_sum_per_col = []
            for W_row_idxs in W_row_idxs_per_col:
                sums = self.W_confidence[W_row_idxs[0]].sum(dim=0)
                neg_sum = torch.roll(self.W_confidence[W_row_idxs[1]].sum(dim=0), shifts = -self.in_dim, dims=0)
                sums += neg_sum
                W_row_idxs_sets_sum_per_col.append(sums)
                
            W_row_idxs_sets_sum_per_col = torch.stack(W_row_idxs_sets_sum_per_col)
            sorted_W_row_idxs_sets_sum_per_col = torch.sort(W_row_idxs_sets_sum_per_col, dim=1, descending=False)

            offset_sorted_W_row_idxs_sets_sum_per_col = sorted_W_row_idxs_sets_sum_per_col.values - sorted_W_row_idxs_sets_sum_per_col.values[:, 0].unsqueeze(1)

            offset_W_row_idxs_sets_sum_to_cols_dict = {}
            for i, offset_sums in enumerate(offset_sorted_W_row_idxs_sets_sum_per_col):
                for offset_sum in offset_sums:
                    if offset_sum.item() not in offset_W_row_idxs_sets_sum_to_cols_dict:
                        offset_W_row_idxs_sets_sum_to_cols_dict[offset_sum.item()] = set()
                    offset_W_row_idxs_sets_sum_to_cols_dict[offset_sum.item()].add(i)
            offset_W_row_idxs_sets_sum_to_cols_dict

            sorted_W_row_idxs_sets_sum = sorted(offset_W_row_idxs_sets_sum_to_cols_dict.keys())

            W_row_idxs_set_sequencing = [offset_W_row_idxs_sets_sum_to_cols_dict[x] for x in sorted_W_row_idxs_sets_sum] # based on increasing offset W row idxs sets sum


            def recursive_fun(W_row_idxs_set_idxs, max_sorted_idx_per_W_row_idxs_set_idxs, used_W_col_idxs):
                if len(W_row_idxs_set_idxs) == 1:
                    W_row_idxs_set_idx = list(W_row_idxs_set_idxs)[0]
                    max_sorted_idx = max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx] if max_sorted_idx_per_W_row_idxs_set_idxs is not None else (self.in_dim * 2) - 1
                    
                    min_sum = None
                    min_sorted_idx = None
                    for i in range(max_sorted_idx + 1):
                        col_idx = sorted_W_row_idxs_sets_sum_per_col.indices[W_row_idxs_set_idx, i].item()
                        if col_idx not in used_W_col_idxs:
                            W_row_idxs_sum = sorted_W_row_idxs_sets_sum_per_col.values[W_row_idxs_set_idx, i].item()
                            if min_sum is None or W_row_idxs_sum < min_sum:
                                min_sum = W_row_idxs_sum
                                min_sorted_idx = i

                    return min_sum, {W_row_idxs_set_idx: min_sorted_idx}

                curr_max_sorted_idx_per_W_row_idxs_set_idxs = [-1] * len(X_row_idxs_per_W_col)

                min_sum = None
                sol_dict = None

                for i in range(len(W_row_idxs_set_sequencing)):
                    offset_W_row_idxs_set_idxs = W_row_idxs_set_sequencing[i]
                    target_W_row_idxs_set_idxs = W_row_idxs_set_idxs & offset_W_row_idxs_set_idxs
                    curr_min_sum = min_sum

                    for W_row_idxs_set_idx in target_W_row_idxs_set_idxs:
                        curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx] += 1
                        if max_sorted_idx_per_W_row_idxs_set_idxs is not None and curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx] > max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]:
                            return min_sum, sol_dict

                        col_idx = sorted_W_row_idxs_sets_sum_per_col.indices[W_row_idxs_set_idx, curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]].item()
                        if col_idx not in used_W_col_idxs:
                            W_row_idxs_sum = sorted_W_row_idxs_sets_sum_per_col.values[W_row_idxs_set_idx, curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]].item()
                            remaining_W_row_idxs_set_idxs = W_row_idxs_set_idxs - {W_row_idxs_set_idx}
                            neg_col_idx = (col_idx + self.in_dim) % (self.in_dim*2) # this might be wrong
                            new_used_col_idxs = used_W_col_idxs | {col_idx, neg_col_idx}
                            nested_sum , sub_sol_dict = recursive_fun(remaining_W_row_idxs_set_idxs, curr_max_sorted_idx_per_W_row_idxs_set_idxs, new_used_col_idxs)

                            if nested_sum is not None:
                                W_row_idxs_sum += nested_sum
                                if min_sum is None or W_row_idxs_sum < min_sum:
                                    min_sum = W_row_idxs_sum
                                    sol_dict = sub_sol_dict
                                    sol_dict[W_row_idxs_set_idx] = curr_max_sorted_idx_per_W_row_idxs_set_idxs[W_row_idxs_set_idx]

                    if min_sum is not None and min_sum == curr_min_sum:
                        return min_sum, sol_dict

                return min_sum, sol_dict

            _, sol_dict = recursive_fun(set(range(len(X_row_idxs_per_W_col))), None, set())
            assert sol_dict is not None
            W_col_to_new_X_row_idxs ={}
            for W_row_idxs_set_idx, sort_idx in sol_dict.items():
                original_col_idx = W_row_idxs_set_idx
                new_col_idx = sorted_W_row_idxs_sets_sum_per_col.indices[W_row_idxs_set_idx, sort_idx].item()
                new_pos_col_idx = new_col_idx if new_col_idx < self.in_dim else new_col_idx - self.in_dim

                original_col = X_row_idxs_per_W_col[original_col_idx]
                if new_pos_col_idx != new_col_idx:
                    original_col = (original_col[1], original_col[0])
                W_col_to_new_X_row_idxs[new_pos_col_idx] = original_col

        adjusted_X_row_idxs_for_zero_Y = {}
        W_row_idxs_with_zero_Ys = [i for i, x in enumerate(one_Y_row_idxs_per_W_row) if len(x) == 0]
        if W_row_idxs_with_zero_Ys:
            available_cols = self.in_dim - len(W_col_to_new_X_row_idxs.keys())
            if available_cols > 0:
                # TODO: this is a problem if you have identical rows of full_X
                X_row_idxs = list(range(self.full_X.shape[0]))
                partitions = random.randint(1, min(available_cols, len(X_row_idxs)))
                selected_partition_idxs = [0]
                if partitions > 1:
                    selected_partition_idxs = sorted(random.sample(set(X_row_idxs) - {0}, partitions - 1))

                last_partition_idx = 0
                X_row_partitions = []
                for partition_idx in (selected_partition_idxs + [len(X_row_idxs)]):
                    row_partition = set(X_row_idxs[last_partition_idx:partition_idx])
                    if row_partition:
                        complement_partition = set(X_row_idxs) - row_partition
                        X_row_partitions.append((complement_partition, row_partition))
                    
                    last_partition_idx = partition_idx

                used_col_idxs = list(W_col_to_new_X_row_idxs.keys())
                neg_used_col_idxs = [ (x + self.in_dim) % (self.in_dim * 2) for x in used_col_idxs]
                available_col_idxs = set(range(self.W.shape[1])) - (set(used_col_idxs) | set(neg_used_col_idxs))
                sums = torch.sort(self.W_confidence[W_row_idxs_with_zero_Ys].sum(dim=0), dim=0, descending=False)

                for col_idx_tensor in sums.indices:
                    idx = col_idx_tensor.item()
                    neg_idx = (idx + self.in_dim) % (self.in_dim * 2)
                    if idx in available_col_idxs and neg_idx not in adjusted_X_row_idxs_for_zero_Y:
                        adjusted_X_row_idxs_for_zero_Y[idx] = X_row_partitions[len(adjusted_X_row_idxs_for_zero_Y.keys())]
                        if len(adjusted_X_row_idxs_for_zero_Y.keys()) == partitions:
                            break
            else:
                def find_best_setup(depth, max_depth, curre_sol, remaining_rows):
                    if depth == max_depth or len(remaining_rows) == 0:
                        return curre_sol if len(remaining_rows) == 0 else None

                    for W_col_idx,X_row_idxs in W_col_to_new_X_row_idxs.items():
                        neg_W_col_idx = (W_col_idx + self.in_dim) % (self.in_dim * 2)
                        sub = remaining_rows - X_row_idxs[1]
                        if W_col_idx not in curre_sol and neg_W_col_idx not in curre_sol and len(sub)< len(remaining_rows):
                            sol = find_best_setup(depth+1, max_depth, curre_sol | {W_col_idx}, sub)
                            if sol is not None:
                                return sol
                        
                        sub = remaining_rows - X_row_idxs[0]
                        if W_col_idx not in curre_sol and neg_W_col_idx not in curre_sol and len(sub)< len(remaining_rows):
                            sol = find_best_setup(depth+1, max_depth, curre_sol | {neg_W_col_idx}, sub)
                            if sol is not None:
                                return sol
                    return None
                
                best_sol = find_best_setup(0, self.in_dim, set(), set(range(self.full_X.shape[0])))
                assert best_sol is not None
                for W_col_idx in best_sol:
                    pos_idx = W_col_idx if W_col_idx < self.in_dim else W_col_idx - self.in_dim
                    target_X_values = W_col_to_new_X_row_idxs[pos_idx]
                    if W_col_idx != pos_idx:
                        target_X_values = (target_X_values[1], target_X_values[0])
                    adjusted_X_row_idxs_for_zero_Y[W_col_idx] = target_X_values

        # END - finding best col config based on W_confidence

        new_W = torch.zeros_like(self.W)
        for W_row_idx, Y_row_idxs in enumerate(one_Y_row_idxs_per_W_row):
            if Y_row_idxs:
                for W_col_idx, X_row_idxs in W_col_to_new_X_row_idxs.items():
                    if Y_row_idxs.issubset(X_row_idxs[0]):
                        new_W[W_row_idx, W_col_idx] = 1
                    elif Y_row_idxs.issubset(X_row_idxs[1]):
                        new_W[W_row_idx, W_col_idx + self.in_dim] = 1

        for row_idx in W_row_idxs_with_zero_Ys:
            new_W[row_idx, list(adjusted_X_row_idxs_for_zero_Y.keys())] = 1

        if not is_first_layer:
            new_full_X = torch.zeros_like(self.full_X)
            for W_col_idx, X_row_idxs in W_col_to_new_X_row_idxs.items():
                new_full_X[list(X_row_idxs[0]), W_col_idx] = 1

            for W_col_idx, X_row_idxs in adjusted_X_row_idxs_for_zero_Y.items():
                target_X_row_idxs = X_row_idxs[0]
                target_complement_X_row_idxs = X_row_idxs[1]
                target_W_col_idx = W_col_idx
                if W_col_idx >= self.in_dim:
                    target_X_row_idxs = X_row_idxs[1]
                    target_complement_X_row_idxs = X_row_idxs[0]
                    target_W_col_idx = W_col_idx - self.in_dim

                new_full_X[list(target_X_row_idxs), target_W_col_idx] = 1
                new_full_X[list(target_complement_X_row_idxs), target_W_col_idx] = 0
        else:
            new_full_X = torch.clone(self.full_X)
    
        self.W = new_W # the problem with this new_W is that it may have rows that are all 0s. There should always be at least one 1 in each row

        return new_full_X[:,:self.in_dim]

class TsetlinMachine:

    def __init__(self, in_dim, clause_dim):
        self.l1 = TsetlinLayer(in_dim, clause_dim)
        self.l2 = TsetlinLayer(clause_dim, clause_dim)
        self.l3 = TsetlinLayer(clause_dim, 1)
        self.out = None

    def forward(self, X):
        X = self.l1.forward(X)
        X = self.l2.forward(X)
        X = self.l3.forward(X)
        self.out = X.squeeze(1)
        return self.out
    
    def update(self, y):
        y = y.unsqueeze(1)
        updated_X = self.l3.update(y)
        updated_X = self.l2.update(updated_X)
        self.l1.update(updated_X, True)