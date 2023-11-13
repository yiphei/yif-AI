import torch
import random

#TODO: need to add a constant random seed

class TsetlinBase:
    def conjunctin_mul(self, X, W):
        matrix_X = X.repeat(W.shape[0], 1)
        mask = W > 0 # TODO: prob need to compare and choose the clause with the highest weight
        masked_X = torch.where(mask, matrix_X, torch.tensor(1))
        return torch.prod(masked_X, dim=1, keepdim=True).view(1,-1)

class TsetlinLayer(TsetlinBase):
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        W_pos = torch.randint(0, 2, (out_dim, in_dim,))
        W_neg = torch.randint(0, 2, (out_dim, in_dim,))
        W_neg[W_pos == 1] = 0
        self.W = torch.cat((W_pos, W_neg), dim=1)
        zero_row_indices = (self.W.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
        col_idx = torch.randint(0, in_dim * 2, (zero_row_indices.shape[0],))
        self.W[zero_row_indices, col_idx] = 1

        self.out = None
        self.full_X = None

    def forward(self, X):
        X_neg = 1 - X
        self.full_X = torch.cat((X, X_neg), dim=0)
        out = self.conjunctin_mul(self.full_X.unsqueeze(0), self.W)
        self.out = out.squeeze(0)
        return self.out
    
    def helper(self, updated_X,update_index, update_W, can_flip_value, can_remove, can_add_value):
        # TODO: the random choice needs to be dynamic, otherwise if it is a very deep layer, it will be very hard to flip values in the earlier layers
        flip_value = random.choice([True, False]) and can_flip_value
        negation_index = (update_index + self.in_dim) % (self.in_dim * 2)
        if flip_value:
            updated_X[update_index] = 1 - updated_X[update_index]
            updated_X[negation_index] = 1 - updated_X[negation_index]

            #TODO: should i set the weight back to 0, as to descrease the confidence of the new flipped clause?
            update_W[update_index] = 1
            update_W[negation_index] = 0
        else:
            addable_indices = [ i for i, (w, v) in enumerate(zip(update_W, updated_X)) if w == 0 and v == 0 and update_W[(i + self.in_dim) % (self.in_dim * 2)] == 0 ] if can_add_value else []

            add = random.choice([True, False]) and len(addable_indices) > 0
            remove = random.choice([True, False]) and can_remove and (update_W.sum() > 1).item()
            if remove:
                update_W[update_index] = 0
            elif add:
                add_index = random.choice(addable_indices)
                update_W[add_index] = 1
            else:
                update_W[update_index] = 0
                update_W[negation_index] = 1

    def update(self, Y, is_first_layer = False):
        can_flip_value = not (is_first_layer or torch.equal(Y, self.out))
        if can_flip_value:
            one_Y_indexes = torch.nonzero(Y == 1).squeeze(1)
            W_halves = torch.split(self.W[one_Y_indexes], self.in_dim, dim=1)
            pos_W = W_halves[0]
            neg_W = W_halves[1]
            for w_1 in pos_W:
                indices = torch.nonzero(w_1 == 1).squeeze(1)
                if any((w_1[indices] == w_2[indices]).any() for w_2 in neg_W):
                    can_flip_value = False
                    break

        updated_X = torch.clone(self.full_X)
        if torch.equal(Y, self.out):
            # TODO: should this be done at every prior layer or should it stop at this layer?
            self.W[self.W > 0] += 1
        else:
            one_Y_indexes = torch.nonzero((Y == 1) & (Y != self.out)).squeeze(1)
            for row_index in one_Y_indexes:
                update_indices = [ i for i, (w, v) in enumerate(zip(self.W[row_index], updated_X)) if w > 0 and v == 0]
                for update_index in update_indices:
                    self.helper(updated_X, update_index, self.W[row_index.item()], can_flip_value, True, False)

            updated_out = self.conjunctin_mul(updated_X.unsqueeze(0), self.W).squeeze(0) if not torch.equal(updated_X, self.full_X) else self.out
            zero_Y_indexes = torch.nonzero((Y == 0) & (Y != updated_out)).squeeze(1)

            for row_index in zero_Y_indexes:
                target_indexes = []
                min_confidence = 0
                for j in range(self.in_dim * 2):
                    W_value = self.W[row_index.item()][j]
                    X_value = updated_X[j]
                    if W_value > 0 and X_value == 1:
                        if W_value < min_confidence or len(target_indexes) == 0:
                            target_indexes = [j]
                            min_confidence = W_value
                        elif W_value == min_confidence:
                            target_indexes.append(j)

                # TODO: there's a bug here when the layer is l1. Fix it
                update_index = random.choice(target_indexes)
                self.helper(updated_X,update_index, self.W[row_index.item()], can_flip_value, False, True)
        return updated_X[:self.in_dim]

class TsetlinMachine:

    def __init__(self, in_dim):
        self.l1 = TsetlinLayer(in_dim, 10)
        self.l2 = TsetlinLayer(10, 1)
        self.out = None

    def forward(self, X):
        X = self.l1.forward(X)
        X = self.l2.forward(X)
        self.out = X.squeeze(0)
        return self.out
    
    def update(self, y):
        y = y.unsqueeze(0)
        updated_X = self.l2.update(y)
        self.l1.update(updated_X, True)