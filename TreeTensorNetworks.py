import numpy as np
from qiskit.quantum_info import SparsePauliOp
import time
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh
import contraction as ctr
import qr_decomposition as qr
import copy
from misc import tlist, loc                                                                                # So that I can index a nested list by passing a tuple as an argument. 
from contextlib import contextmanager
import hamiltonian as hamm

@contextmanager
def timed(section, enabled=True, file=None):
    if not enabled:
        yield
        return
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    file.write(f"{section:<25} {dt:>10.6f}s\n")
    

        
class ttn:
    '''
    A class for a tree tensor networks.
    '''
    def __init__(self, layers, max_bond_dim, local_dim, num_roots = 1, num_child_nodes = 2):

        self.ncn = num_child_nodes                                                                      # Number of child nodes for each node. Default value is 2 (a binary tree).
        self.nroots = num_roots                                                                         # Number of tensors on the top of the tree. Default value is 1.
        self.layers = layers
        self.max_bdim = max_bond_dim                                                                    # Maximal bond dimension for the tree tensor network.
        self.d = local_dim                                                                              # The local physical dimension of each site.
        
        #Derived attributes
        
        self.num = int(num_roots*(self.ncn**layers  - 1)/(-1 + self.ncn))                               # Number of tensors in the binary tree tensor network.
        self.legs = self.ncn + 1                                                                        # Number of legs/links/indices for each node/tensor.
        self.sites = num_roots*self.ncn**(layers+1)                                                     # Number of physical sites (Number of leaf nodes * num_child_nodes) in the ttn = Number of free indices. 


        self.tensors = tlist([[None for j in range(self.nroots * self.ncn**i)] for i in range(self.layers)])           # Is there a better way of storing the tensors? Maybe we could have a tensor class, and a way of instantiating tensors via composition.
        
        
        # Each tensor instance labelled by (or having an attribute of) the layer number (the top two being in layer 0, and the layer number increasing as we go down)
        # and the order the tensor appears in that layer (The leftmost in a layer being numbered 0) in the form of a tuple. This tuple can be called the location of the tensor.  

        # The (layer,position) tuple would give the location and an integer between 0 and self.legs would give the index for each of the contracted indices.


        # Definition : The legs are labelled anti-clockwise from 0 to self.legs - 1 , with the one pointing up (towards to parent node) being 0.

        self.sweep = []                  #This will be the order in which each tensor is optimised in a given sweep.
        for i in range(self.layers):
            for j in range(self.nroots * self.ncn**i):
                self.sweep.append(loc(self,(i,j)))
                                 
    def num_nodes(self,layer):
        '''
        PARAMETERS :
        
        layer   :   The layer index ( = loc[0] ). The topmost layer has layer index = loc[0] = 0.
        
        RETURNS :
        
        The number of tensors in the given layer.
        '''
        if hasattr(layer, "__iter__"):
            #Incase a location tuple is passed instead.
            layer = layer[0]
    
        assert -self.layers <= layer <= self.layers -1, f'{layer} is not a valid layer index for a ttn with {self.layers} layers.'
        return( self.nroots*self.ncn**layer )

    @property
    def num_sites(self):
        return self.nroots * self.ncn**self.layers
    def bdim(self, loc, leg):             #Test this.
                
        '''
        PARAMETERS :
        
        loc   :     Location tuple of a tensor.
        
        leg     :   The index number of the the tensors in that layer (all legs of tensors in a given layer have the same bond dimension).
                    For non-root tensors, the legs are labelled anti-clockwise from 0 to self.legs - 1, with the one pointing towards the parent node being 0.

        RETURNS :
        
        the bond dimension of that leg.
        '''
        layer = loc[0]
        assert -self.layers <= layer <= self.layers -1, f'{layer} is not a valid layer index for a ttn with {self.layers} layers.'

        if layer not in {0,-self.layers} or loc[1] in {0,-1,self.nroots-1,-self.nroots}:
            
            assert leg in range(self.legs), f'{leg} is an invalid leg index for tensors with {self.legs} legs.'
            
        else:
            # Tensors in the root layer can have varying number of indices/legs.
            
            assert leg in range(self.legs + 1), f'{leg} is an invalid leg index for tensors with {self.legs} legs.'
            
            # In this case, the 0 th index points towrads the left root node and the last index points towards the right root node
            
        if self.nroots == 1:        # have to do the case of the root node in a ttn with only one root separtely because that is the only case where you do not have a parent/ root tensor it is connected to
            if layer == 0:
                return min(self.max_bdim, self.d**(self.ncn**(self.layers-1)))

        if leg == 0 or leg == self.ncn+1:                                                                #Pointing towards the parent node.
            return min(self.max_bdim, self.d**(self.ncn**(self.layers - layer)))
        
        else:
            return min(self.max_bdim, self.d**(self.ncn**(self.layers - layer - 1)))
    
    def num_legs(self,loc):
        '''
        Returns the number of legs the tensor at loc has.
        '''
            
        if self.nroots == 1:
            if loc[0] == 0:
                return self.ncn

        if loc[0] != 0 or loc[1] in {0,self.nroots-1}:
            out = self.ncn +1

        else:
            out = self.ncn+2

        return out

    def shape(self,loc):
                
        '''
        PARAMETERS :
        
        loc   :     Location tuple of a tensor.
        
        RETURNS :
        
        A tuple which would be the shape (look at ndarray.shape) of that tensor.
        The convention for naming the indices / legs is the same.
        '''
        return [self.bdim(loc,i) for i in range(self.num_legs(loc))]
    
    def nbd(self, loc1): # This is the heart of any tensor network. It gives out network the structure. There must be a better way of storing this information. 

        # Update this so that you can also pass the bitstring address.

        
        '''
        PARAMETERS :
        
        loc   :   The location tuple of a tensor. 

        RETURNS :
        
        A list containing loc objects wherein self.nbd(loc)[i] = ((i,leg_i)),loc_i).
        Here loc_i is the location of the tensor whose leg_i th index is contracted with the i th index of the tensor at loc. 
        '''

        nb = []
        ad = loc1.adrs
        lyr, pos = loc1[0], loc1[1]
        
        if lyr != 0:    
            #If we are not in the root layer we will have a parent tensor.
            
            if self.nroots !=1:
                parent_leg = int(ad[-1]) + 1 #If there are more than 1 roots, then the 0th index of the parent tensor will be contractd with another root tensor. 
            else:
                parent_leg = int(ad[-1]) + (0 if lyr ==1 else 1)

            nb.insert(0, [parent_leg, loc(self,loc1.parent_location)] )
        
        if lyr == self.layers - 1:
            # If we are in the leaf layer, then we will have no child nodes
            for i in range(len(nb)):
                nb[i][0] = (i,nb[i][0])
            return nb

        # Each tensor will have self.ncn number of child nodes, since the leaf layers is already taken care of.
        for i in range(self.ncn):
            nb.append( [0, loc(self,[lyr+1,pos*self.ncn + i])] )
        
        # Now we need to deal with the root layer.

        if self.nroots == 1:
            for i in range(len(nb)):
                nb[i][0] = (i,nb[i][0])
            return nb
        
        mid_exists = self.nroots > 2
        node_to_the_left_in_mid = pos != 1  # wouldn't a > be more appropriate than a !=

        if lyr == 0: 
            if pos == 0:
                nb.insert(0, [0, loc(self,[0,pos +1 ])])
        
            elif pos == self.nroots -1:
                nb.insert(0, [mid_exists*(self.ncn+1), loc(self, [0,pos - 1])])   # Proud of this line.
        
            #node_to_the_left_in_mid = pos != 1 Can't have this here because it breaks the if-elif chain.
        
            else:
                #Now we must be in mid
                nb.insert(0, [node_to_the_left_in_mid*(self.ncn+1), loc(self,[0,pos-1])] )
                nb.append( [0, loc(self, [0,pos+1])] )
        
        for i in range(len(nb)):
            nb[i][0] = (i,nb[i][0])
        return nb

       # Each tensor will have at most two more legs depending on whether or not it is in the root layer. 


    def randn_init(self):
        # Maybe rewrite this after you change the __add__ method for loc tuples, so that you dont have to use a double for loop. 
        '''
        Initialises all the tensors by assigning real random values to the tensors. 
        '''
        for li in range(len(self.tensors)):
            for pj in range(len(self.tensors[li])):
                locc = loc(self,(li,pj))
                self.tensors[locc] = np.random.randn(*self.shape(locc))       # The * unpacks the tuple.

        # The resultant tensor is not normalised yet.

    def qr(self,loc,leg):
        '''
        Performs the qr decomposition of the tensor at loc with respect the index leg.
        
        PARAMETERS :
        
        loc   :   The location tuple of a tensor.
        leg   :   The leg with respect to you want to isometrise.

        RETURNS :

        A tuple (Q,R)

        Q     :   The isometrised tensor.
        R     :   The non isometric part of the tensor.
        '''

        t = self.tensors[loc]
        q,r = qr.qr_tensors(t,leg)
        return q,r
    
    def iso_index(self,iso_loc,loc): #Test this. 
        '''
        Gives the index along which we would have to qr decompose the tensor at loc 
        if we are isometrising the tn wrt the tensor at iso_loc
        '''
        iso_adrs = iso_loc.adrs
        my_adrs = loc.adrs          # Loc needs to be an object of the loc class

        iso_root_num = int(iso_adrs[0])
        my_root_num = int(my_adrs[0])

        if iso_root_num != my_root_num:   #If we do not originate from the same root
            
            if loc[0] != 0:                 # If the tensor at loc is not in the root layer
                return 0                        #Then go up a layer

            else:                           # If the tensor at loc is in the root layer
                direction = iso_root_num - my_root_num
                # If direction > 0 then move right elif direction <0 move left.
                loc_in_edge = (my_root_num in {0,self.nroots-1})
                if direction < 0 or loc_in_edge:
                    return 0
                else: 
                    # direction can't be 0 because we don't originate from the same root.
                    return self.num_legs(loc) - 1 # This would be the last index for the tensor at loc

        else:                               # We originate from the same root
            if len(iso_adrs) <= len(my_adrs):
                return 0                    # go up a layer if you are in the same layer or in a lower layer than the iso tensor
            
            else:
                if iso_adrs[:len(my_adrs)] == my_adrs:       # If the iso tensor is down the same straight branch as my tensor. 
                    #return int(iso_adrs[len(my_adrs)])  + (self.nroots!=1) # This is the prev version
                    if my_adrs == '0':
                        return int(iso_adrs[len(my_adrs)]) + (self.nroots!=1) 
                    else:   #if loc is not in the root layer
                        return int(iso_adrs[len(my_adrs)]) + 1
                else:
                    return 0 
                    
                 
    @property
    def loc_list(self):
        '''
        Returns a list containing all the possible loc objects for the ttn.
        '''
        loc_list = []
        for layer in range(self.layers):
            for pos in range(self.nroots*self.ncn**layer):
                loc_list.append(loc(self,[layer,pos]))
        return loc_list

    def dist_list(self,my_loc):     # Test this throughouly.
        #can we have a @property decorater to a module if it has more than 1 argument?
        '''
        Creates a list of locations in decreasing order of distance from loc.
        Each element will be a tuple of the form (distance_from_my_loc, location_object)
        '''
        loc_list = self.loc_list
        dist_list = []
        for location in loc_list:
            dist_list.append((my_loc.dist(location),location))
        #now sort dist_list by the first element of each member.
        
        dist_list.sort(key = lambda tup: tup[0], reverse = True)

        return dist_list
    
    @property
    def shape_list(self):
        return [self.shape(l) for l in self.loc_list]
    
    def set_iso(self,iso_loc):
        '''
        Makes the tensor at loc an isometrisation center. Need to implement randn_init first.
        '''
        # Add an assert condition to ensure that self.tensors is not storing None values.
        dist_list = self.dist_list(iso_loc)           
        
        shape_list = self.shape_list
        for item in dist_list[:-1]:     # Excluding the last element because that is iso_loc.
            my_loc = item[1]
            my_iso_index = self.iso_index(iso_loc, my_loc)  # iso_index is the leg pointing towards the iso centre
            my_q, my_r = self.qr(my_loc, my_iso_index)     #Isometrising the tensor wrt the iso_index
            
            assert self.tensors[my_loc].shape == my_q.shape, f'replacement q at {my_loc} does not have the same shape as the tensor it is replacing'
            my_nbd = self.nbd(my_loc)
            legs, next_loc = my_nbd[my_iso_index]
            next_loc_index = legs[1]
            
            self.tensors[my_loc] = my_q                 # Replacing the tensor with the isometrised tensor
            next_tensor = self.tensors[next_loc]
            self.tensors[next_loc] = qr.combine(next_tensor, my_r, next_loc_index, 1)

    def leaf_indices(self,loc):
        '''
        Given a loc, returns the leaf indices that originate
        '''
        return [ loc[1]*self.ncn**(self.layers - loc[0]) + i for i in range(self.ncn**(self.layers- loc[0]))]
    '''
    def optimize_tensor(self, locc, hamiltonian):
        """
        Isometrises the network w.r.t. locc, builds H_eff via effective_network_full,
        finds its ground state, and updates self.tensors[locc].
        """

        self.set_iso(locc)

        env_tensors = ctr.effective_network_full(self, locc)
        H_eff       = ctr.build_H_eff(self, locc, env_tensors, hamiltonian)

        # Symmetrise to guard against floating-point noise
        H_eff = 0.5 * (H_eff + H_eff.T)

        D = H_eff.shape[0]
        if D > 4:
            evals, evecs = eigsh(H_eff, k=1, which='SA')
            ground_vec   = evecs[:, 0]
        else:
            evals, evecs = np.linalg.eigh(H_eff)
            ground_vec   = evecs[:, 0]

        self.tensors[locc] = ground_vec.reshape(self.shape(locc))
        return float(evals[0])

    '''
        

    def local_effective_hamiltonian(self, locc, env_tensors, hamiltonian):
        H_eff = ctr.build_H_eff(self, locc, env_tensors, hamiltonian)
        return H_eff

    def _path_to(self, loc_from, loc_to):
        '''
        gives a list of adjacent locs starting out from loc_from to loc_to.
        '''

        #finding the last common node
        adrs_from = loc_from.adrs
        adrs_to = loc_to.adrs
        
        min_length = min(len(adrs_from), len(adrs_to))
        last_common_node = ''
        for x in range(min_length):
            if adrs_from[x] == adrs_to[x]:
                last_common_node += adrs_from[x]
            else:
                break

        '''
        Create a list containing of all the parent locs starting from loc_from till you reach the last_common_node 
        create another list containing all the parent_locs starting from loc_to till you reach the last common_node and then revrese it
        add them. That will be the path
        '''
        
        last_common_layer = max(len(last_common_node) - 1, 0)   # in case they dont share the same root.    
        path_1 = [loc_from]
        for i in range(loc_from[0] - last_common_layer):
            parent_loc = path_1[-1].parent_location
            path_1.append(parent_loc)

        path_2 = [loc_to]
        for _ in range(loc_to[0] - last_common_layer):
            parent_loc = path_2[-1].parent_location
            path_2.append(parent_loc)
        path_2.reverse()
        
        if last_common_node == '': # if they dont originate from the same root
            path_3 = []             # this will be the movement in the root layer

            direction = np.sign(int(adrs_to[0]) - int(adrs_from[0]))
            for root_pos in range(int(adrs_from[0]), int(adrs_to[0]), direction ):
                path_3.append(loc(self,[0,root_pos]))
        
            path = path_1 + path_3 + path_2
        else:
            path = path_1 + path_2
        # now remove recurring locs, if you wrote the code above better, this would not be necessary. SPEED UP POSSIBLE.
        bag = []
        out = []
        for x in path:      # bag can be a set and this could be a whole lot faster if loc was hashable. Remove this double for loop when you rewrite loc.
            for y in out:
                if x == y:
                    break
            else:
                out.append(x)
        return out

    def shift_isometrisation_centre(self,current_loc, final_loc):
        path = self._path_to(current_loc, final_loc)
       
        for i,my_loc in enumerate(path[:-1]):
            
            my_iso_index = self.iso_index(final_loc, my_loc)  # iso_index is the leg pointing towards the iso centre
            my_q, my_r = self.qr(my_loc, my_iso_index)     #Isometrising the tensor wrt the iso_index
            
            assert self.tensors[my_loc].shape == my_q.shape, f'replacement q at {my_loc} does not have the same shape as the tensor it is replacing'
            my_nbd = self.nbd(my_loc)
            #print(my_loc, my_iso_index)
            legs, next_loc = my_nbd[my_iso_index]
            assert next_loc==path[i+1], f'loc from path {path[i+1]} not matching loc from nbd {next_loc}'
            next_loc_index = legs[1]
            
            #print(f'my loc = {my_loc}, my iso index = {my_iso_index}, next_loc = {next_loc}, next loc index = {next_loc_index}, ')
            #print(f'my shape = {self.tensors[my_loc].shape}, my q shape = {my_q.shape}, my r shape = {my_r.shape}, next tensor shape = {self.tensors[next_loc].shape}\n')
            self.tensors[my_loc] = my_q                 # Replacing the tensor with the isometrised tensor
            next_tensor = self.tensors[next_loc]
            self.tensors[next_loc] = qr.combine(next_tensor, my_r, next_loc_index, 1)
    
    
    def optimize_tensor(self, locc, hamiltonian, level=0):

        #self.set_iso(locc)      # Rewrite this so that the set_iso is used once and then shift_iso(self, loc1, loc2) is used

        env_tensors = ctr.effective_network_full(self, locc)
        print('env_tensors obtained') 
        n_legs    = self.num_legs(locc) # can get rid of these 3 lines
        bond_dims = [self.bdim(locc, leg) for leg in range(n_legs)]
        D         = int(np.prod(bond_dims))


        #H_op = LinearOperator((D, D), matvec=matvec, dtype=float) #what does this do
        H_eff = ctr.build_H_eff(self, locc, env_tensors, hamiltonian)

        print(f'leh obtained {H_eff.shape}')

        # Use current tensor as initial guess — speeds up convergence
        #print(H_eff == H_eff.T)
        if D > 4:
            #v0 = self.tensors[locc].ravel()
            #evals, evecs = eigsh(H_op, k=1, which='SA', v0=v0)
            evals, evecs = np.linalg.eigh(H_eff)     # sparse vs dense, when to use which? Should I just use the eig one instead?
        else:
            # For tiny tensors fall back to dense (eigsh needs D > k+1)
            H_dense = H_op @ np.eye(D)
            H_dense = 0.5 * (H_dense + H_dense.T)
            evals, evecs = np.linalg.eigh(H_dense)

        self.tensors[locc] = evecs[:, level].reshape(self.shape(locc))
        print('tensor optimised')

        # HERE you have to also normalise the tensor at locc.
        return float(evals[level])
    
    def run_sweep(self, hamiltonian, num_sweeps=2, level =0, logging = False, verbose = True, plotting = True,only_final=True, plot_exact=False, log_scale=False):
        '''
        plot_exact : bool
            If True (and log_scale is False), overlays a horizontal reference
            line at the exact ground-state energy (from
            hamiltonian.exact_diagonalization()) on the energy-vs-sweep plot.
        log_scale : bool
            If True, plots log10(|E_exact - E_i|) at every optimisation step
            instead of the raw energy — the usual convergence-curve view,
            whose right-hand plateau shows the residual error the TTN
            settled into. Needs the exact energy either way, so this takes
            priority over plot_exact (which is only meaningful on the raw
            energy plot).
        '''
        energies = []
        sweep = self.sweep
        self.randn_init()
        fh = None
        if logging:
            fh = open(f'{self.num_sites}_sites','a', encoding = 'utf-8')
            fh.write('-'*10+'\n')    # add a way of also logging the run number
            fh.write(f'{self.nroots} roots, {self.ncn} child nodes, {self.layers} layers, max_bond_dim = {self.max_bdim}\t {num_sweeps} sweeps\n\n')
        for sweep_i in range(num_sweeps):
            self.set_iso(loc(self,[0,0]))
            if verbose:
                print(f'\nsweep {sweep_i}\n')
            if logging:
                fh.write(f'\nsweep {sweep_i}\n')
            sweep_energies = []
            for j,locc in enumerate(self.sweep):
                if verbose:
                    print(f'{locc}')

                # sweep[0] is already the isometrisation centre right after set_iso,
                # so there is nothing to shift for j == 0. For every later site we
                # move the centre from the previously-visited site to this one.
                if j > 0:
                    with timed(f'iso_loc_{locc}',logging,fh):
                        self.shift_isometrisation_centre(sweep[j-1],sweep[j])

                        if verbose:
                            print('isometrised')

                with timed('effective_network',logging,fh):
                    env_tensors = ctr.effective_network_full(self, locc)
                    if verbose:
                        print(f'effective_network obtained')

                with timed('build_leh',logging,fh):
                    H_eff = ctr.build_H_eff(self, locc, env_tensors, hamiltonian)
                    if verbose:
                        print(f'local effective hamiltonian obtained, shape : {H_eff.shape}') 

                H_eff = .5* (H_eff + H_eff.T)

                with timed(f'optimise_{H_eff.shape}',logging,fh):
                    evals, evecs = np.linalg.eigh(H_eff)
                    if verbose:
                        print(f'leh optimised')

                # evecs[:, level] is already unit-norm (eigh returns an orthonormal
                # basis); normalising against the *old* tensor that's being
                # overwritten was the bug that broke the canonical form.
                self.tensors[locc] = evecs[:,level].reshape(self.shape(locc))

                sweep_energies.append(evals[level])

                if verbose:
                    print(f'\nSweep {sweep_i + 1:3d}, loc {locc}: E = {evals[0]:.8f}\n')
            if verbose:
                print('\n')
            energies.append(sweep_energies)
        if logging:
            fh.write(f'energies = {energies}\n')
            fh.close()
        
        if plotting:
            unravelled_energies = []
            num_tensors = len(self.sweep)
            for i, sweep_en in enumerate(energies):
                unravelled_energies += sweep_en

            unravelled_energies = np.array(unravelled_energies)
            steps    = np.arange(len(unravelled_energies))
            final_en = unravelled_energies[-1]

            exact_en = None
            if plot_exact or log_scale:
                exact_en = hamiltonian.exact_diagonalization()

            fig, ax = plt.subplots(figsize=(8, 5), dpi=110)

            # Sweep-boundary guides (skip x=0, it just duplicates the axis)
            for i in range(1, num_sweeps):
                ax.axvline(x=i * num_tensors, ls='--', color='0.75', lw=1, zorder=1)

            if log_scale:
                errors = np.abs(exact_en - unravelled_energies)
                tiny   = np.finfo(float).tiny
                errors = np.where(errors == 0, tiny, errors)   # guard log(0)
                y      = np.log10(errors)

                ax.plot(steps, y, color='#3366CC', lw=1.5, zorder=2)
                ax.scatter(steps, y, color='#3366CC', s=20, zorder=3,
                           edgecolor='white', linewidth=0.6)

                final_err     = max(abs(exact_en - final_en), tiny)
                final_log_err = np.log10(final_err)
                ax.axhline(y=final_log_err, ls='--', color='k', lw=1,
                           label=f'Final error = {final_err:.2e}', zorder=1)

                ax.set_ylabel(r'$\log_{10}|E_{\mathrm{exact}} - E|$')
                ax.set_title(f'TTN convergence to exact energy\n'
                             f'({self.num_sites} sites, '
                             fr'$\chi_{{\mathrm{{max}}}}$ = {self.max_bdim})')
                ax.legend(frameon=False, loc='upper right')

            else:
                ax.plot(steps, unravelled_energies, color='#3366CC', lw=1.5,
                        zorder=2, label='TTN energy')
                ax.scatter(steps, unravelled_energies, color='#3366CC', s=20,
                           zorder=3, edgecolor='white', linewidth=0.6)

                ax.axhline(y=final_en, ls='--', color='k', lw=1,
                           label=f'Final energy = {final_en:.6f}', zorder=1)

                if plot_exact:
                    ax.axhline(y=exact_en, ls='-', color='#CC3333', lw=1.3,
                               alpha=0.9, label=f'Exact energy = {exact_en:.6f}',
                               zorder=1)

                ax.set_ylabel('Energy')
                ax.set_title(f'TTN sweep convergence\n'
                             f'({self.num_sites} sites, '
                             fr'$\chi_{{\mathrm{{max}}}}$ = {self.max_bdim})')
                ax.legend(frameon=False, loc='best')

            ax.set_xlabel('Sweep')
            ax.set_xticks(np.arange(num_sweeps * num_tensors, step=num_tensors))
            ax.set_xticklabels(np.arange(num_sweeps))
            ax.set_xlim(steps[0], steps[-1])
            ax.grid(alpha=0.3, lw=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            fig.tight_layout()
            plt.show()
        if only_final:
            return energies[-1][-1]
        return energies
    
    def _quiet_run_sweep(self, hamiltonian, num_sweeps):
        '''
        the same as run_sweep without the bells and whistles. 
        mostly for debugging. Remove it when done. 
        Maybe we can break the bigger function down into more readable bits, like moving the plotting to a different method entirely.
        '''
        sweep = self.sweep
        self.randn_init()

        # grnd state sweep
        energies = []
        for sweep_i in range(num_sweeps):
            self.set_iso(loc(self, [0,0]))
            sweep_energies = []

            for j, locc in enumerate(self.sweep):
                
                #print(f'sweep {sweep_i} optimising loc {locc}')
                env_tensors = ctr.effective_network_full(self, locc)
                H_eff = ctr.build_H_eff(self, locc, env_tensors, hamiltonian)

                H_eff = .5*(H_eff + H_eff.T)

                evals, evecs = np.linalg.eigh(H_eff)
                
                self.tensors[locc] = evecs[:,0].reshape(self.shape(locc))
                norm = np.linalg.norm(self.tensors[locc])
                self.tensors[locc] = self.tensors[locc] / norm
                
                if j!=len(sweep)-1:
                    self.shift_isometrisation_centre(sweep[j],sweep[j+1])

                sweep_energies.append(evals[0])

            energies.append(sweep_energies)
            #print(f'\nsweep {sweep_i} over')
        
        """
        unravelled_energies = []
        num_tensors = len(self.sweep)
        for i, sweep_en in enumerate(energies):
            unravelled_energies += sweep_en

        plt.plot(range(len(unravelled_energies)), unravelled_energies)
        plt.scatter(range(len(unravelled_energies)), unravelled_energies,label='Energies from sweep')
        plt.title(f'Number of sites = {self.num_sites} and $\\chi_{{\\mathrm{{max}}}} = {self.max_bdim}$') 
        min_en = min(unravelled_energies)
        max_en = max(unravelled_energies)

        final_en = unravelled_energies[-1]
        for i in range(num_sweeps+1):
            plt.axvline(x = i*num_tensors ,linestyle = '--', color='k')

        exact_energy = hamiltonian.exact_diagonalization(only_ground=False)[0]
        
        plt.axhline(y = exact_energy, ls='--',color='k',label='Exact energy')
        plt.grid()
        plt.xticks(np.arange((num_sweeps)*len(self.sweep), step=len(self.sweep)),np.arange(1,num_sweeps+1))
        plt.xlabel('Sweeps')
        plt.ylabel('Energy')
        plt.legend(loc='upper right') 
        plt.show()
        """
        return [en[-1] for en in energies]

        
    
    def run_sweepp(self, hamiltonian, num_sweeps=1, verbose=True):
        """
        Main variational sweep. For each loc in self.sweep: isometrise → build H_eff
        → optimise tensor. Repeats for num_sweeps full passes.
        Returns list of energies (one per sweep, estimated from the last site's eigenvalue).
        """
        energies = []
        for sweep_idx in range(num_sweeps):
            #last_energy = None
            for locc in self.sweep:
                print(locc)
                last_energy = self.optimize_tensor(locc, hamiltonian)
            energies.append(last_energy)
            if verbose:
                print(f'Sweep {sweep_idx + 1:3d}:  E = {last_energy:.10f}')
        return energies

    
    def flush(self):
        '''
        empties out all the tensors in the self.tensors list
        '''
        self.tensors = tlist([[None for j in range(self.nroots * self.ncn**i)] for i in range(self.layers)])           

    def __repr__(self):
        return(f'Number of roots = {self.nroots}\nNumber of child nodes = {self.ncn}\nNumber of layers = {self.layers}')
        
        
'''
TO DO:
1. Check if the network if being normalised after optimisation (is that even important?)
2. Get unbalanced ttns working.
3. Integrate MultiSiteOperator

4. For some reason, run_sweep is working better than run_sweepp, 
   even though run sweep is just a riced up version of run_swepp. Look into why that is, and remove the duplicate.

5. Note that the method ttn.optimize_tensor is not actually being used in ttn.run_sweep.
   That bit is written again, hardcoded into ttn.run_sweep.
   See if ttn.optimize_tensor is working how it should be (there is one that is commented out, check that as well. ), and then use the method, trimming the amount of code you have.

'''

    
ham2 = hamm.TFIsing(8,n_legs = 8,periodic=False)
ham4 = hamm.TFIsing(4,n_legs = 4,periodic=False)
hamf = hamm.Free(8)
ham3 = hamm.TFIsing(16, n_legs = 16,periodic = True)
#ham3 = hamm.TFIsing()

ttn1 = ttn(layers = 2, max_bond_dim = 7, local_dim = 2, num_child_nodes = 2, num_roots = 5)
ttn2 = ttn(layers = 4, max_bond_dim = 7, local_dim = 2, num_child_nodes = 2, num_roots = 1)
ttn3 = ttn(layers = 3, max_bond_dim = 5, local_dim = 2, num_child_nodes = 2, num_roots = 2)
ttn4 = ttn(layers = 2, max_bond_dim = 3, local_dim = 2, num_child_nodes = 2, num_roots = 2)
ttn5 = ttn(layers = 1, max_bond_dim = 3, local_dim = 2, num_child_nodes = 2, num_roots = 2)

