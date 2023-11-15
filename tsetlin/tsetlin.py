import torch
import random

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
    
    def helper(self, expected_X,update_idx, expected_W, can_flip_value, can_remove, can_add_value):
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

    def update(self, Y, is_first_layer = False):
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
        if torch.equal(Y, self.out):
            # TODO: should this be done at every prior layer or should it stop at this layer?
            self.W[self.W > 0] += 1
        else:
            for single_Y, single_out, single_expected_X, single_can_flip_value in zip(Y, self.out, expected_X, can_flip_value):
                one_Y_idxs = torch.nonzero((single_Y == 1) & (single_Y != single_out)).squeeze(1)
                for row_idx in one_Y_idxs:
                    update_idxs = [ i for i, (w, v) in enumerate(zip(self.W[row_idx], single_expected_X)) if w > 0 and v == 0]
                    for update_idx in update_idxs:
                        self.helper(single_expected_X, update_idx, self.W[row_idx.item()], single_can_flip_value, True, False)

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
                    self.helper(single_expected_X,update_idx, self.W[row_idx.item()], single_can_flip_value, False, True)
        return expected_X[:,:self.in_dim]

class TsetlinMachine:

    def __init__(self, in_dim):
        clause_dim = 10
        self.l1 = TsetlinLayer(in_dim, clause_dim)
        self.l2 = TsetlinLayer(clause_dim, 1)
        self.out = None

    def forward(self, X):
        X = self.l1.forward(X)
        X = self.l2.forward(X)
        self.out = X.squeeze(1)
        return self.out
    
    def update(self, y):
        y = y.unsqueeze(1)
        updated_X = self.l2.update(y)
        self.l1.update(updated_X, True)