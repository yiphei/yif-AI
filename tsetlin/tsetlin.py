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
    
    def old_helper(self, expected_X,update_idx, expected_W, can_flip_value, can_remove, can_add_value):
        # TODO: the random choice needs to be dynamic, otherwise if it is a very deep layer, it will be very hard to flip values in the earlier layers
        should_flip_value = random.choice([True, False]) and can_flip_value
        negation_idx = (update_idx + self.in_dim) % (self.in_dim * 2)
        if should_flip_value:
            expected_X[update_idx] = 1 - expected_X[update_idx]
            expected_X[negation_idx] = 1 - expected_X[negation_idx]

            #TODO: should i set the weight back to 0, as to descrease the confidence of the new flipped clause?
            expected_W[update_idx] = 1
            expected_W[negation_idx] = 0
        else:
            addable_idxs = [ i for i, (w, x) in enumerate(zip(expected_W, expected_X)) if w == 0 and x == 0 and expected_W[(i + self.in_dim) % (self.in_dim * 2)] == 0 ] if can_add_value else []

            should_add = random.choice([True, False]) and len(addable_idxs) > 0
            should_remove = random.choice([True, False]) and can_remove and (expected_W > 1).sum().item() > 1
            if should_remove:
                expected_W[update_idx] = 0
            elif should_add:
                add_index = random.choice(addable_idxs)
                expected_W[add_index] = 1
            else:
                expected_W[update_idx] = 0
                expected_W[negation_idx] = 1

    def old_update(self, Y, is_first_layer = False):
        can_flip_value = torch.full((Y.shape[0],), False)
        if not is_first_layer:
            for i, (single_Y, single_out) in enumerate(zip(Y, self.out)):
                if not torch.equal(single_Y, single_out):
                    can_flip_value[i] = True

                    one_Y_idxs = torch.nonzero(single_Y == 1).squeeze(1)
                    W_halves = torch.split(self.W[one_Y_idxs], self.in_dim, dim=1)
                    pos_W = W_halves[0]
                    neg_W = W_halves[1]
                    for w_1 in pos_W:
                        idxs = torch.nonzero(w_1 == 1).squeeze(1)
                        if any((w_1[idxs] == w_2[idxs]).any() for w_2 in neg_W):
                            can_flip_value[i] = False
                            break
        
        expected_X = torch.clone(self.full_X)
        can_modify_W = torch.full((self.W.shape[0],), True)
        for i in range(self.W.shape[0]):
            one_Y_idxs = torch.nonzero(Y[:,i] == 1).squeeze(1)
            target_X = expected_X[one_Y_idxs]
            one_intersections = target_X.prod(dim=0, keepdim=True).squeeze(0)
            one_intersection_idxs = torch.nonzero(one_intersections == 1).squeeze(1)
            if len(one_intersection_idxs) == 0:
                can_modify_W[i] = False

        if torch.equal(Y, self.out):
            # TODO: should this be done at every prior layer or should it stop at this layer?
            self.W[self.W > 0] += 1
        else:
            for single_Y, single_out, single_expected_X, single_can_flip_value in zip(Y, self.out, expected_X, can_flip_value):
                one_Y_idxs = torch.nonzero((single_Y == 1) & (single_Y != single_out)).squeeze(1)
                for row_idx in one_Y_idxs:
                    update_idxs = [ i for i, (w, v) in enumerate(zip(self.W[row_idx], single_expected_X)) if w > 0 and v == 0]
                    for update_idx in update_idxs:
                        self.old_helper(single_expected_X, update_idx, self.W[row_idx.item()], single_can_flip_value, True, False)

            updated_out = self.conjunction_mul(expected_X.unsqueeze(1), self.W) if not torch.equal(expected_X, self.full_X) else self.out
            for single_Y, single_out, single_expected_X, single_can_flip_value in zip(Y, updated_out, expected_X, can_flip_value):
                zero_Y_idxs = torch.nonzero((single_Y == 0) & (single_Y != single_out)).squeeze(1)
                for row_idx in zero_Y_idxs:
                    candidate_idxs = []
                    min_W = 0
                    for j in range(self.in_dim * 2):
                        W_value = self.W[row_idx.item()][j]
                        X_value = single_expected_X[j]
                        if W_value > 0 and X_value == 1:
                            if W_value < min_W or len(candidate_idxs) == 0:
                                candidate_idxs = [j]
                                min_W = W_value
                            elif W_value == min_W:
                                candidate_idxs.append(j)

                    update_idx = random.choice(candidate_idxs)
                    self.old_helper(single_expected_X,update_idx, self.W[row_idx.item()], single_can_flip_value, False, True)
        return expected_X[:,:self.in_dim]


    def update(self, Y, is_first_layer = False):
        if torch.equal(Y, self.out):
            return self.full_X[:,:self.in_dim]

        zero_Y_row_idxs_per_W_row = []
        one_Y_row_idxs_per_W_row = []
        for i in range(self.W.shape[0]):
            row_Y = Y[:, i]
            
            zero_Y_idxs = torch.nonzero(row_Y == 0).squeeze(1).tolist()
            zero_Y_row_idxs_per_W_row.append(set(zero_Y_idxs))

            one_Y_idxs = torch.nonzero(row_Y == 1).squeeze(1).tolist()
            one_Y_row_idxs_per_W_row.append(set(one_Y_idxs))

        update_fnc = self.update_batch_first_layar if is_first_layer else self.update_batch_non_first_layer
        return update_fnc(one_Y_row_idxs_per_W_row, zero_Y_row_idxs_per_W_row)


    def update_batch_first_layar(self, one_Y_row_idxs_per_W_row, _):
        one_Y_idxs_to_W_row_idx = {}
        for i, x in enumerate(one_Y_row_idxs_per_W_row):
            if x:
                if tuple(x) not in one_Y_idxs_to_W_row_idx:
                    one_Y_idxs_to_W_row_idx[tuple(x)] = []
                one_Y_idxs_to_W_row_idx[tuple(x)].append(i)

        new_W = torch.zeros_like(self.W)

        for col_idx in range(self.full_X.shape[1]//2):
            one_idxs = set((self.full_X[:, col_idx] == 1).nonzero().squeeze(1).tolist())
            zero_idxs = set((self.full_X[:, col_idx] == 0).nonzero().squeeze(1).tolist())

            subsets_of_one_idxs = [x for x in one_Y_row_idxs_per_W_row if x and x.issubset(one_idxs)]
            subsets_of_zero_idxs = [x for x in one_Y_row_idxs_per_W_row if x and x.issubset(zero_idxs)]

            for subset_of_one_idxs in subsets_of_one_idxs:
                new_W[one_Y_idxs_to_W_row_idx[tuple(subset_of_one_idxs)], col_idx] = 1
            for subset_of_zero_idxs in subsets_of_zero_idxs:
                opposite_col_idx = (col_idx + self.in_dim) % (2 * self.in_dim)
                new_W[one_Y_idxs_to_W_row_idx[tuple(subset_of_zero_idxs)], opposite_col_idx] = 1

        self.W = new_W
        return None


    def update_batch_non_first_layer(self, one_Y_row_idxs_per_W_row, zero_Y_row_idxs_per_W_row):
        unique_one_Y_row_idxs = set()
        visited_ones = set()
        for i,x in enumerate(one_Y_row_idxs_per_W_row):
            if x:
                tuple_x = tuple(x)
                if tuple_x not in visited_ones:
                    visited_ones.add(tuple_x)
                    unique_one_Y_row_idxs.add(i)


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
                    left_W = curr_one_Y_idxs | opposite_remaining_Y_subset
                    right_W = min_zero_Y_subset | remaining_Y_subset

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
                    
                    #add remaining with the curr_clause
                    left_W = curr_one_Y_idxs | remaining_Y_subset
                    right_W = min_zero_Y_subset | opposite_remaining_Y_subset

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
        
        cols, solved = recursive_helper(0, self.in_dim, tracking, q.popleft(), q)
        assert solved

        new_W = torch.zeros_like(self.W)
        for row_idx, x in enumerate(one_Y_row_idxs_per_W_row):
            for i, col in enumerate(cols):
                    col_left = col[0]
                    col_right = col[1]
                    if x.issubset(col_left):
                        new_W[row_idx, i] = 1
                    elif x.issubset(col_right):
                        new_W[row_idx, i + self.in_dim] = 1
        self.W = new_W

        new_full_X = torch.zeros_like(self.full_X)
        for i, col in enumerate(cols):
            new_full_X[list(col[0]), i] = 1
        
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