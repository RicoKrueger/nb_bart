#from joblib import Parallel, delayed
from numba import jit
import sys
import time

import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2

@jit(nopython=True)
def check_unique(x):
    """ Checks if there are at least two unique values in each column of x. 
    
    Arguments: 
        x (array): Array of dimension N x J
        
    Returns
        u (array): J-dimensional arrray of booleans indicating whether the 
        corresponding column in x contains at least two unique values. 
    
    """
    
    N, J = x.shape
    u = np.zeros((J,))
    for j in np.arange(J):
        v0 = x[0,j]
        for n in np.arange(1,N):
            if x[n,j] != v0:
                u[j] = 1
                break
    return u
    
class Data:
    """ Parent class for DataTrain and DataVal. """
    def __init__(self, X):
        self.X = X
        self.N, self.J = X.shape
        
class DataTrain(Data):
    """ Holds the training data.
    
    Attributes:
        y_raw (array): Dependent variables on original scale
        y (array): Scaled dependent variables to be used for model training.
        w (array): Observation-specific weights used for heteroskedastic model.
        offset (float): offset shifting data such that y -> y - offset is used 
        for training.
        y_raw_mid: Midpoint of y_raw (needed for unscaling)
        y_raw_ran: Range of y_raw (needed for unscaling)
    
    """      
    
    def __init__(self, y_raw, X, weights=0, offset=0):
        self.y_raw = y_raw
        super().__init__(X)
        self.offset = offset
        self.weights = weights
        
        y_raw_off = y_raw - offset
        y_raw_off_min = np.amin(y_raw_off)
        y_raw_off_max = np.amax(y_raw_off)
        self.y_raw_off_mid = ((y_raw_off_min + y_raw_off_max) / 2)
        self.y_raw_off_ran = y_raw_off_max - y_raw_off_min
        
        self.y = (y_raw_off - self.y_raw_off_mid) / self.y_raw_off_ran
        
    def update(self, y_raw_new, weights_new):
        """ Updates dependent data and weights """
        self.y = (y_raw_new - self.offset - self.y_raw_off_mid) \
            / self.y_raw_off_ran
        self.weights = weights_new
            
    def unscale(self, y):
        """ Transforms dependent data to original scale. """
        return y * self.y_raw_off_ran + self.y_raw_off_mid + self.offset

class DataVal(Data):
    """ Holds the validation data """
    def __init__(self, y, X):
        self.y = y
        super().__init__(X)  
        
class BartOptions:
    """ Contains BART options. 
    
    Attributes:
        data: Training data object.
        nTrees: Number of trees.
        k: Determines prior probability that E(Y|X) is contained within
        [y_min, y_max] based on normal distribution.
        sigma_mu: Prior probability that E(Y|X) is contained within
        [y_min, y_max] based on normal distribution.
        nu: Degrees of freedom of Inverse-Chi^2 prior on error variance.
        q: Quantile of prior on error variance.
        sigma_emp: Empirical estimate of error scale.
        lam: Scale of Inverse-Chi^2 prior on error variance. 
        alpha: Base hyperparameter of tree prior.
        beta: Power hyperparameter of tree prior.
        p_grow: Prior probability of grow mutation.
        p_prune: Prior probability of prune mutation.
        p_change: Prior probability of change mutation.
        heter: Boolean indicating whether model is heteroskedastic
        extract_variable_inclusion_props: Boolean indicating whether variable
        inclusion proportions should be extracted.
        N: Number of observations.
        J: Number of predictors.
    
    """
    
    @staticmethod
    def _set_lam(data, nu, q):
        """ Sets lambda hyperparameter and calculates empirical error 
        scale. 
        
        Arguments:
            data: Trainig data object.
            nu: nu hyperparameter.
            q: Quantile
            
        Returns:
            sigma_emp: Empirical estimate of error scale.
            lam: lambda hyperparameter.
        """
        
        #Estimate empirical standard deviation
        if data.J < data.N:
            lm = sm.OLS(data.y, sm.add_constant(data.X)).fit() 
            sigma_emp = np.sqrt(lm.ssr / (data.N - lm.df_model - 1))
        else:
            sigma_emp = np.std(data.y)
        
        #Set lambda
        qchi = chi2.ppf(1 - q, nu)
        lam = qchi * sigma_emp**2 / nu
        return sigma_emp, lam
    
    def __init__(
            self, 
            data,
            nTrees=200, 
            k=2.0, nu=3.0, q=0.9,
            alpha=0.95, beta=2.0,
            p_grow=0.28, p_prune=0.28, p_change=0.44,
            estimate_sigma=True, heter=False,
            extract_variable_inclusion_props=True
            ):
        
        assert (p_grow + p_prune + p_change) == 1, \
            'Mutation probabilities do not add up to one!'
        
        if heter:
            assert data.weights is not None, \
                'Weights need to be provided for heteroskedastic model.'
        
        self.nTrees = nTrees
        self.k = k
        self.sigma_mu = 1 / (2 * k * np.sqrt(nTrees))
        self.nu = nu
        self.q = q
        if estimate_sigma:
            self.sigma_emp, self.lam = self._set_lam(data, nu, q)
        else:
            self.sigma_emp = 1.0 / data.y_raw_off_ran
            self.lam = None
        self.alpha = alpha
        self.beta = beta
        self.p_grow = p_grow
        self.p_prune = p_prune
        self.p_change = p_change
        self.estimate_sigma = estimate_sigma
        self.heter = heter
        self.N = data.N
        self.J = data.J
        
class Split:
    """ Represents the split at a node.
    
    Arguments:
        variable (int): Index of the variable on which the split is performed.
        condition (int): Split condition. data < condition is sent to the left
        child; data >= condition is sent to the right child. 
        
    """
    
    def __init__(self, variable, condition):
        self.variable = variable
        self.condition = condition  
        
    def left_split(self, data, data_idx):
        """ Performs left split
        
        Arguments:
            data (DataTrain): Training data.
            data_idx (array): Logical indices which indicate which rows in the
            data are associated with the parent node.
        
        Returns:
            new_data_idx (array): Boolean indices which indicate which rows in
            the data are associated with the current node.
            unique_x (array): Boolean indices which indicate in which rows of 
            the design matrix contain at least two unique values.
        
        """            
        new_data_idx = np.logical_and(
            data_idx, data.X[:,self.variable] < self.condition) 
        unique_x = check_unique(data.X[new_data_idx,:])
        return new_data_idx, unique_x
        
    def right_split(self, data, data_idx):
        """ Performs right split (analogous to left split). """
        new_data_idx = np.logical_and(
            data_idx, data.X[:,self.variable] >= self.condition) 
        unique_x = check_unique(data.X[new_data_idx,:])
        return new_data_idx, unique_x

class Node:
    """ Represents tree node (both internal node and leaf node)
    
    Attributes:
        data_idx: Boolean indices indicating which rows of the data are 
        associated with the node
        unique_x: J-dimensional Array of booleans indicating whether there are
        at least two distinct values of the j-th predictor after the split 
        associate with the node.
        mu: Leaf parameters.
        parent: Parent node.
        left_child: Left child node.
        right_child: Righ child node.
        
    """
    
    def __init__(self, 
                 data_idx, unique_x=None, mu=0,
                 parent=None, split=None, left_child=None, right_child=None):
        self.data_idx = data_idx
        self.unique_x = unique_x
        self.mu = mu
        self.split = split
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        
    def deep_copy(self, parent=None):
        """ Creates a fast deep copy of the node. """
        node_copy = Node(self.data_idx, self.unique_x, self.mu)
        node_copy.parent = parent
        node_copy.split = self.split
        if self.is_leaf():
            return [node_copy]
        else:  
            node_copy_left_children = self.left_child.deep_copy(node_copy)
            node_copy_right_children = self.right_child.deep_copy(node_copy)
            node_copy.left_child = node_copy_left_children[0]
            node_copy.right_child = node_copy_right_children[0]
            return [node_copy, 
                    *node_copy_left_children, *node_copy_right_children]
        
    def depth(self):
        """ Returns depth of node (node location relative to root node) """
        if self.parent is None:
            d = 0
        else:
            d = 1 + self.parent.depth()
        return d
    
    def height(self):
        """ Returns height of node (node location relative to deepest leaf 
        node. """
        if (self.left_child is None) and (self.right_child is None):
            return 0
        else:
            return 1 + max(self.left_child.height(), 
                           self.right_child.height())       
        
    def is_leaf(self):
        """ Checks if node is leaf node. """
        return (self.left_child is None) and (self.right_child is None)
        
    def branch(self):
        """ Returns the current node and the branch below the current node. 
        """
        if self.is_leaf():
            return [self]
        else:
            return [self, 
                    *self.left_child.branch(), *self.right_child.branch()]
        
    def become_leaf(self):
        """ Transforms (internal) node into leaf node. """
        self.split = None
        self.left_child = None
        self.right_child = None 
        
    def check_children_empty(self):
        """ Checks if children are not associated with any observations. """
        return np.sum(self.left_child.data_idx) \
            * np.sum(self.right_child.data_idx) == 0 
        
    def get_responses(self, data):
        """ Returns responses associated with the node, if the node is a 
        leaf node. """
        if self.is_leaf():
            return np.array(data.y[self.data_idx])
        else:
            raise Exception('Not a leaf node!')
            
    def get_residuals(self, data, y_hat):
        """ Returns residuals associated with the node, if the node is a 
        leaf node. """
        if self.is_leaf():
            resid = np.array(data.y[self.data_idx] - y_hat[self.data_idx] \
                             + self.mu)
            return resid
        else:
            raise Exception('Not a leaf node!')
            
    def get_weights(self, weights):
        """ Returns weights associated with the node, if the node is a 
        leaf node. """
        if self.is_leaf():
            return np.array(weights[self.data_idx])
        else:
            raise Exception('Not a leaf node!')
        
    def update_leaf_parameter(self, sigma, sigma_mu, data, y_hat):
        """ Draws posterior samples of the leaf parameters. """
        resid = self.get_residuals(data, y_hat)
        nResid = resid.shape[0]
        denom = 1 / sigma_mu**2 + nResid / sigma**2
        mu_mu = (np.sum(resid) / sigma**2) / denom
        mu_si = np.sqrt(1 / denom)
        self.mu = mu_mu + mu_si * np.random.randn()
        
    def update_leaf_parameter_heter(
            self, sigma, sigma_mu, data, y_hat):
        """ Draws posterior samples of the leaf parameters for the 
        heteroskedastic model. """
        resid = self.get_residuals(data, y_hat)
        w = self.get_weights(data.weights)
        ws2 = (w * sigma)**2
        denom = 1 / sigma_mu**2 + np.sum(1 / ws2)
        mu_mu = np.sum(resid / ws2) / denom
        mu_si = np.sqrt(1 / denom)
        self.mu = mu_mu + mu_si * np.random.randn()
    
    def predict(self, X):
        """ Performs out of sample prediction for one observation
        
        Arguments:
            X (array): Predictors associated with the observation.
            
        Returns:
            Leaf parameters conditional on X
            
        """
        
        if self.is_leaf():
            return self.mu
        elif X[self.split.variable] < self.split.condition:
            return self.left_child.predict(X)
        else:
            return self.right_child.predict(X)
        

class Tree:
    """ Represents a tree made up of nodes.
    
    Attributes:
        bart_options: BART options
        root_node: Root node (top node)
        nodes: List of all nodes in the tree
    
    """
    
    ###
    #Definition and essential functions
    ###    
    
    def __init__(self, bart_options, root_node, nodes=None):
        self.bart_options = bart_options
        self.root_node = root_node
        if nodes is None:
            self.nodes = [self.root_node]
        else:
            self.nodes = nodes
        
    def deep_copy(self):
        """ Creates a fast deep copy of a tree. """
        nodes_copy = self.root_node.deep_copy()
        tree_copy = Tree(self.bart_options, nodes_copy[0], nodes_copy)
        return tree_copy        
        
    def max_depth(self):
        """ Returns maximum depth of the tree. """
        return max([n.depth() for n in self.nodes])
    
    def height(self):
        """ Returns height of the tree. """
        return self.root_node.height()

    def update_nodes(self):
        """ Updates the list of nodes based on parent-children 
        relationship. """
        self.nodes = self.root_node.branch()
        return self.nodes
    
    def leaf_nodes(self):
        """ Returns all leaf nodes in the tree. """
        return [n for n in self.nodes if n.is_leaf()] 
    
    def internal_nodes(self):
        """ Returns all internal nodes in the tree. """
        return [n for n in self.nodes if not n.is_leaf()] 
    
    def singly_internal_nodes(self):
        """ Returns all singly internal nodes (nodes with exactly two terminal 
        children) in the tree. """
        return [n for n in self.nodes if n.height()==1]
    
    def fit(self):
        """ Fits training data to tree. 
            
        Returns:
            y_hat: Fitted values
        
        """
        y_hat = np.zeros((self.bart_options.N,))
        for n in self.leaf_nodes():
            y_hat[n.data_idx] = n.mu
        return y_hat
    
    def predict(self, data_val):
        """ Performs out-of-sample prediction.
        
        Arguments: 
            data_val: Validation data object.
            
        Returns:
            y_pred: predicted values for each observation in data_val.X
            
        """
        
        y_pred = np.zeros((data_val.N,))
        for n in np.arange(data_val.N):
            y_pred[n] = self.root_node.predict(data_val.X[n,:])
        return y_pred
    
    ###
    #Tree mutation
    ###
    
    def grow(self, node, data, split):
        """ Grows the tree by adding children to the given node. """
        #Update split
        node.split = split
        
        #Add left child
        data_idx, unique_x = split.left_split(data, node.data_idx)
        node.left_child = Node(data_idx, unique_x, node.mu, node)
        
        #Add right child
        data_idx, unique_x = split.right_split(data, node.data_idx)
        node.right_child = Node(data_idx, unique_x, node.mu, node)
        
        #Update nodes
        self.update_nodes()
    
    def prune(self, node):
        """ Prunes the tree by removing children from the given node. """
        #Remove children from node
        node.become_leaf() 
        
        #Update nodes
        self.update_nodes()
        
    def change(self, node, data, split): 
        """ Changes tree by changing decision rule of the given node. """
        #Update split
        node.split = split
        
        #Update left child
        data_idx, unique_x = split.left_split(data, node.data_idx)
        node.left_child.data_idx = data_idx
        node.left_child.unique_x = unique_x
        node.left_child.mu = node.mu
        
        #Update right child
        data_idx, unique_x = split.right_split(data, node.data_idx)
        node.right_child.data_idx = data_idx
        node.right_child.unique_x = unique_x
        node.right_child.mu = node.mu

    ###
    #Tree update
    ###
        
    def _grow_log_transition_ratio(self, b, J_adj, j_ux, w2_star):
        """ Calculates the log transition ratio for the grow proposal. """
        log_r = np.log(self.bart_options.p_prune) \
            - np.log(self.bart_options.p_grow) \
            + np.log(b) + np.log(J_adj) + np.log(j_ux) - np.log(w2_star)
        return log_r
    
    def _grow_log_likelihood_ratio(self, node, node_star, sigma, data, y_hat):
        """ Calculates the log likelihood ratio for the grow proposal. """
        s2 = sigma**2
        sm2 = self.bart_options.sigma_mu**2
        
        resid = node.get_residuals(data, y_hat)
        resid_l = node_star.left_child.get_residuals(data, y_hat)
        resid_r = node_star.right_child.get_residuals(data, y_hat)
        
        nResid = resid.shape[0]
        nResid_l = resid_l.shape[0]
        nResid_r = resid_r.shape[0]
        
        denom = s2 + nResid * sm2
        denom_l = s2 + nResid_l * sm2
        denom_r = s2 + nResid_r * sm2
        
        log_r0 = np.log(sigma) \
            + 0.5 * ( np.log(denom) \
                     -np.log(denom_l) \
                     -np.log(denom_r))
        log_r10 = sm2 / 2 / s2
        log_r11  = np.sum(resid_l)**2 / denom_l
        log_r11 += np.sum(resid_r)**2 / denom_r
        log_r11 -= np.sum(resid)**2 / denom
        log_r = log_r0 + log_r10 * log_r11
        return log_r
    
    def _grow_log_likelihood_ratio_heter(
            self, node, node_star, sigma, data, y_hat):
        """ Calculates the log likelihood ratio for the grow proposal for the
        heteroskedastic model."""
        w = node.get_weights(data.weights)
        w_l = node_star.left_child.get_weights(data.weights)
        w_r = node_star.right_child.get_weights(data.weights)
        
        ws2 = (w * sigma)**2
        ws2_l = (w_l * sigma)**2
        ws2_r = (w_r * sigma)**2
        
        sm2 = self.bart_options.sigma_mu**2
        
        resid = node.get_residuals(data, y_hat)
        resid_l = node_star.left_child.get_residuals(data, y_hat)
        resid_r = node_star.right_child.get_residuals(data, y_hat)
        
        denom = 1 + sm2 * np.sum(1 / ws2)
        denom_l = 1 + sm2 * np.sum(1 / ws2_l)
        denom_r = 1 + sm2 * np.sum(1 / ws2_r)
        
        log_r0 = 0.5 * (np.log(denom) - np.log(denom_l) - np.log(denom_r))
        log_r10 = sm2 / 2
        log_r11  = np.sum(resid_l / ws2_l)**2 / denom_l
        log_r11 += np.sum(resid_r / ws2_r)**2 / denom_r
        log_r11 -= np.sum(resid / ws2)**2 / denom
        log_r = log_r0 + log_r10 * log_r11
        return log_r    
        
    def _grow_log_tree_structure_ratio(self, d, J_adj, j_ux):
        """ Calculates the log tree structure ratio for the grow proposal. """
        alpha = self.bart_options.alpha
        beta = self.bart_options.beta 
        log_r = np.log(alpha) + 2 * np.log(1 - alpha / (2+d)**beta) \
            - np.log((1+d)**beta - alpha) - np.log(J_adj) - np.log(j_ux)
        return log_r
    
    def _grow_propose(self, sigma, data, y_hat):
        """ Makes a grow proposal. 
        
        Arguments:
            sigma: Scale of the error term
            data: Training data object.
            
        Returns:
            tree_star: the mutated tree (proposal)
            log_alpha: log acceptance ratio
        """
        #Select leaf node to split on
        leaf_nodes = self.leaf_nodes()
        b = len(leaf_nodes)
        eta_node = leaf_nodes[np.random.choice(b)]
        eta_node_id = self.nodes.index(eta_node)
        
        #Select attribute to split on
        j = np.random.choice(self.bart_options.J)
        if eta_node.unique_x[j] == 0:
            return None, None
        J_adj = np.sum(eta_node.unique_x)
            
        #Mutate tree by growing tree
        split_data = data.X[eta_node.data_idx,j]
        split_data_unique = np.unique(split_data)
        j_ux = split_data_unique.shape[0]
        split_value = np.random.choice(split_data_unique)
        split = Split(j, split_value)
        tree_star = self.deep_copy()
        eta_node_star = tree_star.nodes[eta_node_id]
        tree_star.grow(eta_node_star, data, split)
        
        #Check if children contain data
        if eta_node_star.check_children_empty():
            return None, None
        
        #Calculate log transition ratio
        w2_star = len(tree_star.singly_internal_nodes())     
        log_transition_ratio = self._grow_log_transition_ratio(
            b, J_adj, j_ux, w2_star)
        
        #Calculate log likelihood ratio
        if not self.bart_options.heter:       
            log_likelihood_ratio = self._grow_log_likelihood_ratio(
                eta_node, eta_node_star, sigma, data, y_hat)
        else:
            log_likelihood_ratio = self._grow_log_likelihood_ratio_heter(
                eta_node, eta_node_star, sigma, data, y_hat)
        
        #Calculate log tree structure ratio
        log_tree_structure_ratio = self._grow_log_tree_structure_ratio(
            eta_node.depth(), J_adj, j_ux)
            
        #log alpha
        log_alpha = log_transition_ratio \
            + log_likelihood_ratio \
            + log_tree_structure_ratio
            
        return tree_star, log_alpha
    
    def _prune_log_transition_ratio(self, b, J_adj, j_ux, w2):
        """ Calculates the log transition ratio for the prune proposal. """
        log_r = np.log(self.bart_options.p_grow) \
            - np.log(self.bart_options.p_prune) \
            + np.log(w2) - np.log(b-1) - np.log(J_adj) - np.log(j_ux)
        return log_r   
    
    def _prune_propose(self, sigma, data, y_hat): 
        """ Makes a prune proposal. 
        
        Arguments:
            sigma: Scale of the error term
            data: Training data object.
            
        Returns:
            tree_star: the mutated tree (proposal)
            log_alpha: log acceptance ratio
        """
        #Select singly internal node without any grandchildren to split on
        singly_internal_nodes = self.singly_internal_nodes()
        w2 = len(singly_internal_nodes)
        if w2 == 0:
            return None, None
        eta_node = singly_internal_nodes[np.random.choice(w2)]
        eta_node_id = self.nodes.index(eta_node)
        b = len(self.leaf_nodes())
        
        #Attribute which is split on
        j = eta_node.split.variable
        j_ux = np.unique(data.X[eta_node.data_idx,j]).shape[0]
        J_adj = np.sum(eta_node.unique_x)
        
        #Mutate tree by pruning tree
        tree_star = self.deep_copy()
        eta_node_star = tree_star.nodes[eta_node_id]
        tree_star.prune(eta_node_star)
        
        #Calculate log transition ratio   
        log_transition_ratio = self._prune_log_transition_ratio(
            b, J_adj, j_ux, w2)
        
        #Calculate log likelihood ratio
        if not self.bart_options.heter:       
            log_likelihood_ratio = -self._grow_log_likelihood_ratio(
                eta_node_star, eta_node, sigma, data, y_hat)
        else:
            log_likelihood_ratio = -self._grow_log_likelihood_ratio_heter(
                eta_node_star, eta_node, sigma, data, y_hat) 
        
        #Calculate log tree structure ratio
        log_tree_structure_ratio = -self._grow_log_tree_structure_ratio(
            eta_node.depth(), J_adj, j_ux)
        
        #log alpha
        log_alpha = log_transition_ratio \
            + log_likelihood_ratio \
            + log_tree_structure_ratio      
        return tree_star, log_alpha
    
    def _change_log_likelihood_ratio(
            self, node, node_star, sigma, data, y_hat):
        """ Calculates the log likelihood ratio for the change proposal. """
        s2 = sigma**2
        sm2 = self.bart_options.sigma_mu**2
        vr = s2 / sm2 
        
        resid_l = node.left_child.get_residuals(data, y_hat)
        resid_r = node.right_child.get_residuals(data, y_hat)
        
        resid_l_star = node_star.left_child.get_residuals(data, y_hat)
        resid_r_star = node_star.right_child.get_residuals(data, y_hat)
        
        nResid_l = resid_l.shape[0]
        nResid_r = resid_r.shape[0]
        nResid_l_star = resid_l_star.shape[0]
        nResid_r_star = resid_r_star.shape[0]
        
        vn_l = vr + nResid_l
        vn_r = vr + nResid_r    
        vn_l_star = vr + nResid_l_star
        vn_r_star = vr + nResid_r_star        
        
        log_r0 = 0.5 * ( np.log(vn_l) + np.log(vn_r)
                        -np.log(vn_l_star) - np.log(vn_r_star))
        log_r10 = 1 / 2 / s2
        log_r11  = np.sum(resid_l_star)**2 / vn_l_star
        log_r11 += np.sum(resid_r_star)**2 / vn_r_star
        log_r11 -= np.sum(resid_l)**2 / vn_l
        log_r11 -= np.sum(resid_r)**2 / vn_r
        log_r = log_r0 + log_r10 * log_r11
        return log_r

    def _change_log_likelihood_ratio_heter(
            self, node, node_star, sigma, data, y_hat):
        """ Calculates the log likelihood ratio for the change proposal for
        the heteroskedastic model."""
        w_l = node.left_child.get_weights(data.weights)
        w_r = node.right_child.get_weights(data.weights)
        w_l_star = node_star.left_child.get_weights(data.weights)
        w_r_star = node_star.right_child.get_weights(data.weights)
        
        ws2_l = (w_l * sigma)**2
        ws2_r = (w_r * sigma)**2
        ws2_l_star = (w_l_star * sigma)**2
        ws2_r_star = (w_r_star * sigma)**2
  
        sm2 = self.bart_options.sigma_mu**2
        
        resid_l = node.left_child.get_residuals(data, y_hat)
        resid_r = node.right_child.get_residuals(data, y_hat)
        resid_l_star = node_star.left_child.get_residuals(data, y_hat)
        resid_r_star = node_star.right_child.get_residuals(data, y_hat)
                
        denom_l = 1 + sm2 * np.sum(1 / ws2_l) 
        denom_r = 1 + sm2 * np.sum(1 / ws2_r) 
        denom_l_star = 1 + sm2 * np.sum(1 / ws2_l_star) 
        denom_r_star = 1 + sm2 * np.sum(1 / ws2_r_star) 
        
        log_r0 = 0.5 * ( np.log(denom_l) + np.log(denom_r)
                        -np.log(denom_l_star) - np.log(denom_r_star))
        log_r10 = sm2 / 2
        log_r11  = np.sum(resid_l_star / ws2_l_star)**2 / denom_l_star
        log_r11 += np.sum(resid_r_star / ws2_r_star)**2 / denom_r_star
        log_r11 -= np.sum(resid_l / ws2_l)**2 / denom_l
        log_r11 -= np.sum(resid_r / ws2_r)**2 / denom_r
        log_r = log_r0 + log_r10 * log_r11
        return log_r
    
    def _change_propose(self, sigma, data, y_hat):
        """ Makes a change proposal. 
        
        Arguments:
            sigma: Scale of the error term
            data: Training data object.
            y_hat (array): The sum of trees 
            
        Returns:
            tree_star: the mutated tree (proposal)
            log_alpha: log acceptance ratio
        """
        #Select singly internal node (no grandchildren) to split on
        singly_internal_nodes = self.singly_internal_nodes()
        w2 = len(singly_internal_nodes)
        if w2 == 0:
            return None, None
        eta_node = singly_internal_nodes[np.random.choice(w2)]
        eta_node_id = self.nodes.index(eta_node)
        
        #Select new attribute to split on
        j_cand = np.asarray(np.nonzero(eta_node.unique_x)).reshape(-1,)
        j = np.random.choice(j_cand)
           
        #Mutate tree by changing node
        split_data = data.X[eta_node.data_idx,j]
        split_data_unique = np.unique(split_data)
        split_value = np.random.choice(split_data_unique)
        split = Split(j, split_value)
        tree_star = self.deep_copy()
        eta_node_star = tree_star.nodes[eta_node_id]
        tree_star.change(eta_node_star, data, split)
        
        #Check if children contain data
        if eta_node_star.check_children_empty():
            return None, None
        
        #Calculate log likelihood ratio
        if not self.bart_options.heter:
            log_likelihood_ratio = self._change_log_likelihood_ratio(
                eta_node, eta_node_star, sigma, data, y_hat)
        else:
            log_likelihood_ratio = self._change_log_likelihood_ratio_heter(
                eta_node, eta_node_star, sigma, data, y_hat)
        
        #log alpha
        log_alpha = log_likelihood_ratio
            
        return tree_star, log_alpha
        
    def _update_tree(self, sigma, data, y_hat):
        """ Performs a Metropolis-Hastings step to update the tree by 
        mutation. """
        #Propose new tree
        u = np.random.rand()
        if u <= self.bart_options.p_grow:
            tree_star, log_alpha = self._grow_propose(sigma, data, y_hat)
        elif u <= (self.bart_options.p_grow + self.bart_options.p_prune):
            tree_star, log_alpha = self._prune_propose(sigma, data, y_hat)
        else:
            tree_star, log_alpha = self._change_propose(sigma, data, y_hat)
            
        if tree_star is None:
            return False
        
        #Metropolis-Hastings
        log_r = np.log(np.random.rand())            
        if log_r <= log_alpha:
            self.root_node = tree_star.root_node
            self.nodes = tree_star.nodes
            return True
        else:
            return False
        
    def _update_leaf_parameters(self, sigma, data, y_hat):
        """ Updates the leaf parameters of the tree. """
        sigma_mu = self.bart_options.sigma_mu
        for n in self.leaf_nodes():
            if not self.bart_options.heter:
                n.update_leaf_parameter(sigma, sigma_mu, data, y_hat)
            else:
                n.update_leaf_parameter_heter(
                    sigma, sigma_mu, data, y_hat)
            
    def update(self, sigma, data, y_hat):
        """ Updates the tree by mutating it and by updating its leaf 
        parameters. """
        proposal_accepted = self._update_tree(sigma, data, y_hat)
        self._update_leaf_parameters(sigma, data, y_hat)    
        return proposal_accepted
        
class Forest:  
    """ Represents a forest made up of trees. 
    
    Attributes:
        trees: List of trees.
        sigma: Scale of error term distribution (eps ~ N(0, sigma^2))
        nTrees: Number of trees in the forest
        bart_options: BART options
        y_hat (array): Fitted values 
        
    """
    
    def __init__(self, bart_options, data):
        self.trees = []
        self.sigma = bart_options.sigma_emp
        self.nTrees = bart_options.nTrees
        self.bart_options = bart_options
        self.y_hat = np.zeros((self.bart_options.N))
        
        for i in np.arange(self.nTrees):
            root = Node(np.ones((self.bart_options.N,), dtype=bool),
                        check_unique(data.X))
            tree = Tree(bart_options, root)
            self.trees.append(tree)  
            
    def _update_sigma(self, y, y_hat, data):
        """ Draws posterior sample of the scale of the error term. """
        if not self.bart_options.heter:
            ssr = np.sum((y - y_hat)**2)
        else:
            ssr = np.sum(((y - y_hat) / data.weights)**2)
        df = self.bart_options.nu + self.bart_options.N
        scale_df = (self.bart_options.nu * self.bart_options.lam + ssr)     
        self.sigma = np.sqrt(scale_df / np.random.chisquare(df))        
        
    def update(self, data):
        """ Updates the forest by updating all trees and the scale. """
        #Update trees and fit
        mean_tree_depth = 0
        mean_tree_accepted = 0
        
        for t, tree in enumerate(self.trees):
            y_hat_tree = tree.fit()
            proposal_accepted = tree.update(self.sigma, data, self.y_hat)
            self.y_hat += tree.fit() - y_hat_tree
             
            mean_tree_accepted += proposal_accepted
            mean_tree_depth += tree.max_depth()
        
        mean_tree_accepted /= self.nTrees
        mean_tree_depth /= self.nTrees
        
        #Update sigma
        if self.bart_options.estimate_sigma:
            self._update_sigma(data.y, self.y_hat, data)
 
        return mean_tree_accepted, mean_tree_depth
    
    def predict(self, data_val):
        """ Performs out-sample prediction as sum of all trees. """
        y_pred = 0
        for t in self.trees:
            y_pred += t.predict(data_val)
        return y_pred
    
    def variable_inclusion(self):
        """ Extracts variable inclusion proportions for forest. """
        variable_inclusion_props = np.zeros((self.bart_options.J,))
        for t in self.trees:
            for n in t.internal_nodes():
                variable_inclusion_props[n.split.variable] += 1
        variable_inclusion_props /= variable_inclusion_props.sum()
        return variable_inclusion_props
                    
class ChainDraws():
    """ Holds the posterior draws of a chain. """
    
    def __init__(
            self, 
            chain_id, data,
            y_hat_draws, sigma_draws, y_pred_draws, 
            avg_tree_acceptance, avg_tree_depth,
            variable_inclusion_props_draws
            ):
        self.chain_id = chain_id
        self.y_hat_draws = data.unscale(y_hat_draws)
        self.sigma_draws = sigma_draws
        self.y_pred_draws = data.unscale(y_pred_draws)
        self.avg_tree_acceptance = avg_tree_acceptance
        self.avg_tree_depth = avg_tree_depth
        self.variable_inclusion_props_draws = variable_inclusion_props_draws
        
class McmcResults():
    """ Holds the results of a MCMC simulation.
    
    Attributes:
        mcmc_options: MCMC options.
        time: estimation time.
        
        y_hat_draws: Draws of fitted values.
        sigma_draws: Draws of error scale.
        y_pred_draws: Draws of out-of-sample predictions.
        avg_tree_acceptance_draws: Draws of avg. tree acceptance proportions.
        avg_tree_depth_draws: Draws of avg. tree depth.
        variable_inclusion_props_draws: Draws of variable inclusion 
        proportions.
        
        y_hat_mean: Posterior means of fitted values.
        sigma_mean: Posterior mean of error scale.
        y_pred_mean: Posterior mean of out-of-sample predictions.
        
        avg_tree_acceptance_mean: Posterior means of avg. tree acceptance 
        proportions.
        avg_tree_depth_mean: Posterior means of avg. tree depth.
    
    """
    
    def __init__(self, mcmc_options, toc, draws):
        self.mcmc_options = mcmc_options
        self.time = toc
        
        self.y_hat_draws = np.stack([d.y_hat_draws for d in draws], axis=0)
        self.sigma_draws = np.stack([d.sigma_draws for d in draws], axis=0)
        self.y_pred_draws = np.stack([d.y_pred_draws for d in draws], axis=0)
        self.avg_tree_acceptance_draws \
            = np.stack([d.avg_tree_acceptance for d in draws], axis=0)
        self.avg_tree_depth_draws \
            = np.stack([d.avg_tree_depth for d in draws], axis=0)
        self.variable_inclusion_props_draws \
            = np.stack([d.variable_inclusion_props_draws for d in draws], 
                       axis=0)
        
        self.y_hat_mean = self.y_hat_draws.mean(axis=(0,1))
        self.sigma_mean = self.sigma_draws.mean(axis=(0,1))
        self.y_pred_mean = self.y_pred_draws.mean(axis=(0,1))
        self.avg_tree_acceptance_mean = self.avg_tree_acceptance_draws.mean()
        self.avg_tree_depth_mean = self.avg_tree_depth_draws.mean()
        
class McmcOptions:
    """ Contains options for MCMC simulation of BART.
    
    Attributes (in addition to parent class):
        nChain: Number of MCMC chains.
        nBurn: Number of burn-in samples.
        nSample: Number of samples after burn-in period.
        nIter: Number of total samples.
        nThin: Thinning
        nKeep: Number of samples to keep after burn-in, given thinning
        mcmc_disp: After how many iterations to print simulation progress.        
    
    """
    def __init__(
            self, 
            nChain=1, nBurn=400, nSample=400, nThin=1, mcmc_disp=100,
            ):
        self.nChain = nChain
        self.nBurn = nBurn
        self.nSample = nSample
        self.nIter = nBurn + nSample
        self.nThin = nThin
        self.nKeep = int(nSample / nThin)
        self.mcmc_disp = mcmc_disp
    
class BartMcmc:
    """ Performs MCMC simulation of BART. 
    
    Attributes:
        mcmc_options: MCMC options
        bart_options: BART options
        data: Training data object
        data_val: Validation data object 
        
    """    
    
    def __init__(self, mcmc_options, bart_options, data, data_val):
        self.mcmc_options = mcmc_options
        self.bart_options = bart_options
        self.data = data
        self.data_val = data_val
    
    def _mcmc_chain(self, chain_id, data, data_val):
        """ One MCMC chain.
        
        Arguments:
            chain_id (int): identifier of the chain
            data: Training data object
            data_val: Validation data object
            
        Returns
            chain_draws: An instance of the class MmcmcResults containing the 
            posterior samples of the chain.
        
        """
        
        ###
        #Storage
        ###
        
        y_hat_draws = np.zeros((self.mcmc_options.nKeep, data.N))
        sigma_draws = np.zeros((self.mcmc_options.nKeep,))
        y_pred_draws = np.zeros((self.mcmc_options.nKeep, data_val.N))
        avg_tree_acceptance = np.zeros((mcmc_options.nIter,))
        avg_tree_depth = np.zeros((mcmc_options.nIter,))
        variable_inclusion_props_draws = np.zeros((self.mcmc_options.nKeep, 
                                                   data.J))
        
        ###
        #Initialise
        ###
        
        forest = Forest(self.bart_options, data)
        
        ###
        #Sampling
        ###
        
        j = -1
        sample_state = 'burn in' 
        
        for i in np.arange(self.mcmc_options.nIter):  
            #Posterior updates
            avg_tree_acceptance[i], avg_tree_depth[i] = forest.update(data)
            
            #Display progress
            if ((i+1) % self.mcmc_options.mcmc_disp) == 0:
                if (i+1) > self.mcmc_options.nBurn:
                    sample_state = 'sampling'
                    
                sl = slice(max(i+1-100,0),i+1)
                ravg_depth = np.round(np.mean(avg_tree_depth[sl]), 2)
                ravg_acceptance = np.round(np.mean(avg_tree_acceptance[sl]), 2)
                
                print('Chain ' + str(chain_id) + 
                      '; iteration: ' + str(i+1) + ' (' + sample_state + ')'
                      '; avg. tree depth: ' + str(ravg_depth) + 
                      '; avg. tree acceptance: ' + str(ravg_acceptance) +
                      ';')
                sys.stdout.flush()           
            
            if (i+1) > self.mcmc_options.nBurn \
                and ((i+1) % self.mcmc_options.nThin)==0:
                    j += 1
                    y_hat_draws[j,:] = forest.y_hat
                    sigma_draws[j] = forest.sigma * data.y_raw_off_ran
                    y_pred_draws[j,:] = forest.predict(data_val)
                    variable_inclusion_props_draws[j,:] \
                        = forest.variable_inclusion()
        
        #Return
        chain_draws = ChainDraws(chain_id, data,
                                 y_hat_draws, sigma_draws, y_pred_draws, 
                                 avg_tree_acceptance, avg_tree_depth,
                                 variable_inclusion_props_draws)           
        return chain_draws
    
    def estimate(self):
        """ Estimates BART via MCMC. """
        tic = time.time()
        draws = [self._mcmc_chain(c, self.data, self.data_val) 
                 for c in range(self.mcmc_options.nChain)]
        """
        draws = Parallel(n_jobs=self.mcmc_options.nChain)\
            (delayed(self._mcmc_chain)\
             (c, self.data, self.data_val) 
             for c in range(self.mcmc_options.nChain))
        """
        toc = time.time() - tic
        print('Estimation time [s]:' + str(toc))
        results = McmcResults(self.mcmc_options, toc, draws)
        return results
    
class SyntheticData:
    """ Generates synthetic data. 
    
    Attributes:
        N_train (int): Number of training samples.
        N_val (int): Number of validation samples. 
        
    """
    
    def __init__(self, N_train=100, N_val=10):
        self.N_train = N_train
        self.N_val = N_val
        self.N = N_train + N_val
    
    @staticmethod
    def _friedman_f(x):
        """ Friedman's 5-dimensional test function. """
        f = 10 * np.sin(np.pi * x[:,0] * x[:,1]) + 20 * (x[:,2] - 0.5)**2 \
        + 10 * x[:,3] + 5 * x[:,4]
        return f
    
    def _split(self, y, x):
        """ Splits data in training and validation data. """
        y_train = np.array(y[:self.N_train])
        x_train = np.array(x[:self.N_train,:])
        y_val = np.array(y[self.N_train:])
        x_val = np.array(x[self.N_train:,:])
        return (y_train, x_train), (y_val, x_val)
        
    def friedman(self, p=10):
        """ Generates continuous responses based on Friedman's test function.
        
        Arguments: 
            p (int): Number of predictors (must be at least five).
            
        Returns:
            synth: Tuple of training and validation data.
        """
        assert p >= 5, 'p must be at least 5.'
        x = np.random.rand(self.N, p)
        f_x = self._friedman_f(x)
        eps = np.random.randn(self.N)
        y = f_x + eps
        return self._split(y, x)
    
    def friedman_heter(self, p=10):
        """ Generates continuous responses based on Friedman's test function 
        with heteroskedastic error.
        
        Arguments: 
            p (int): Number of predictors (must be at least five).
            
        Returns:
            synth: Tuple of training and validation data.
            weights_train: array of weights for the training sample
        """
        assert p >= 5, 'p must be at least 5.'
        x = np.random.rand(self.N, p)
        x_idx = np.random.choice(p, size=int(p/2), replace=False)
        u = np.random.rand(int(p/2),)
        x[:,x_idx] = x[:,x_idx] <= u
        
        f_x = self._friedman_f(x)
        eps = np.random.randn(self.N)
        
        r = 0.2 + 0.8 * np.random.rand(self.N_train,)
        weights_train = r / r.sum() * self.N_train
        weights = np.ones((self.N,))
        weights[:self.N_train] = weights_train
        
        offset = 0
        
        y = f_x + weights * eps + offset
        
        synth = self._split(y, x)
        
        return synth, weights_train
    

if __name__ == "__main__":
    
    np.random.seed(4711)
    
    ###
    #Generate data
    ###
    
    heter = False
    
    if not heter:
        synth = SyntheticData(N_train=500, N_val=50).friedman(50)        
        data = DataTrain(*synth[0])
        data_val = DataVal(*synth[1])
        
        np.savetxt('data.csv', np.hstack(
            (data.y_raw.reshape(-1,1), data.X)), delimiter=',')
        np.savetxt('data_val.csv', np.hstack(
            (data_val.y.reshape(-1,1), data_val.X)), delimiter=',')
                
        bart_options = BartOptions(data)
        mcmc_options = McmcOptions()
    else:        
        synth, weights_train = SyntheticData(N_train=500, N_val=20)\
        .friedman_heter()
        data = DataTrain(*synth[0], weights_train, offset=0)
        data_val = DataVal(*synth[1])
        
        np.savetxt('data.csv', np.hstack(
            (data.y_raw.reshape(-1,1), data.X, 
             weights_train.reshape(-1,1))), delimiter=',')
        np.savetxt('data_val.csv', np.hstack(
            (data_val.y.reshape(-1,1), data_val.X)), delimiter=',')
    
        bart_options = BartOptions(data, heter=True)
        mcmc_options = McmcOptions()
    
    ###
    #Estimate
    ###

    results = BartMcmc(mcmc_options, bart_options, data, data_val).estimate()
    rmse = np.sqrt(np.mean((data.y_raw - results.y_hat_mean)**2))
    rmse_test = np.sqrt(np.mean((data_val.y - results.y_pred_mean)**2))
    print(rmse)
    print(rmse_test)
    
    np.corrcoef(data.y_raw, results.y_hat_mean, rowvar=False)
    np.corrcoef(data_val.y, results.y_pred_mean, rowvar=False)
    
    ###
    #Variable importance
    ###
    
    import matplotlib.pyplot as plt
    #import seaborn as sns
    
    x = np.arange(1, data.J+1)
    var_inclu = results.variable_inclusion_props_draws
    var_inclu_mean = var_inclu.mean(axis=(0,1))
    var_inclu_q = np.abs(np.quantile(var_inclu, axis=(0,1), q=(0.025, 0.975)) \
        - var_inclu_mean)
    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    ax = plt.errorbar(x=x, y=var_inclu_mean, yerr=var_inclu_q, fmt='o')
    