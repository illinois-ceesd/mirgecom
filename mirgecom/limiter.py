""":mod:`mirgecom.limiter` is for limiters and limiter-related constructs.
Field limiter functions
-----------------------
.. autofunction:: limiter_liu_osher
"""

__copyright__ = """
Copyright (C) 2022 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from functools import partial

from arraycontext import map_array_container
from meshmode.dof_array import DOFArray
from grudge.discretization import DiscretizationCollection
import grudge.op as op
import numpy as np

from arraycontext import thaw, freeze

def neighbor_list(dim, mesh):
    adj = mesh.facial_adjacency_groups[0]
    nconnections = (adj[0].elements).shape[0]
    connections = np.empty((nconnections, 2), dtype=np.int32)

    #putting in a ordered sequence
    connections[:,0] = np.sort(adj[0].elements)
    connections[:,1] = adj[0].neighbors[np.argsort(adj[0].elements)]

    neighbors = np.empty((mesh.nelements,dim+2),dtype=np.int32)
    ii = 0
    for kk in range(0,mesh.nelements):
      neighbors[kk,:] = kk
      #neighbors[kk, 0] = kk
      #neighbors[kk,1:] = connections[ii,1]
      idx = 0
      while connections[ii,0] == kk:
        idx += 1
        neighbors[kk,idx] = connections[ii,1]
        ii += 1

        if (ii == nconnections):
          break

    return neighbors

def cell_volume(dcoll: DiscretizationCollection, field):
    return op.elementwise_integral(dcoll, field*0.0 + 1.0)

def drop_order(dcoll: DiscretizationCollection, volume, field):
    actx = field.array_context

    # Compute cell averages of the state
    cell_avgs = 1.0/volume*op.elementwise_integral(dcoll, field)
    avgs = actx.to_numpy(cell_avgs[0])[:,0]

    return 0.0*(field - cell_avgs) + cell_avgs


#def limiter_liu_osher(dcoll: DiscretizationCollection, neig, volume, field):
#    """.

#    Parameters
#    ----------
#    dcoll: :class:`grudge.discretization.DiscretizationCollection`
#        Grudge discretization with boundaries object
#    field: meshmode.dof_array.DOFArray or numpy.ndarray
#        A field or collection of scalar fields to limit
#    Returns
#    -------
#    meshmode.dof_array.DOFArray or numpy.ndarray
#        An array container containing the limited field(s).
#    """
#    
#    actx = field.array_context
#    
#    # Compute cell averages of the state
#    cell_avgs = 1.0/volume*op.elementwise_integral(dcoll, field)     
#    avgs = actx.to_numpy(cell_avgs[0])[:,0]

#    # Compute nodal and elementwise max/mins of the field
#    mmax_i = actx.to_numpy(op.elementwise_max(dcoll, field)[0])[:,0]
#    mmin_i = actx.to_numpy(op.elementwise_min(dcoll, field)[0])[:,0]
#        
##    # Cell gradient
##    grad_i = op.local_grad(dcoll, field)
##    grad_X = 1.0/volume*op.elementwise_integral(dcoll, grad_i[0])
##    grad_Y = 1.0/volume*op.elementwise_integral(dcoll, grad_i[1])    

##    grad = np.sqrt( (actx.to_numpy( grad_X[0] )[:,0])**2 + 
##                    (actx.to_numpy( grad_Y[0] )[:,0])**2 )
#        
#    # Compute minmod factor (Eq. 2.9)
#    nneighbors = neig.shape[1]

#    mmax = np.maximum( avgs[neig[:,0]], avgs[neig[:,1]] )
#    for i in range(1,nneighbors):
#        mmax[:] = np.maximum( mmax[:], avgs[neig[:,i]] )
#    
#    mmin = np.minimum( avgs[neig[:,0]], avgs[neig[:,1]] )
#    for i in range(1,nneighbors):
#        mmin[:] = np.minimum( mmin[:], avgs[neig[:,i]] )
#        
##    #including neighbors of neighbors reduces dissipation
##    for i in range(1,nneighbors):
##        mmax[:] = np.maximum( mmax[:], mmax[neig[:,i]] )
##    for i in range(1,nneighbors):
##        mmin[:] = np.minimum( mmin[:], mmin[neig[:,i]] )
#    
#    _theta = np.minimum(
#                1., np.minimum(
#                abs( (mmax-avgs)/(mmax_i-avgs+1e-12) ),
#                abs( (mmin-avgs)/(mmin_i-avgs+1e-12) ) )
#             )

##    #conditional to bypass the boundary cells
##    #_theta = np.where(neig[:,1] == neig[:,0],1.0,_theta)
##    _theta = np.where(neig[:,2] == neig[:,0],1.0,_theta)
##    _theta = np.where(neig[:,3] == neig[:,0],1.0,_theta)

#    # Transform back to array context
#    #FIXME apparently there is a broadcast operation
#    dummy = np.zeros(cell_avgs[0].shape)
#    for i in range(0,cell_avgs[0].shape[-1]):
#      dummy[:,i] = _theta[:]

#    theta = DOFArray(actx, data=(actx.from_numpy(np.array(dummy)), ))
#  
#    return theta*(field - cell_avgs) + cell_avgs

def positivity_preserving_limiter(dcoll: DiscretizationCollection, volume, field):
    """Implement the positivity-preserving limiter of Liu and Osher (1996).
    """

    actx = field.array_context

    # Compute cell averages of the state
    cell_avgs = 1.0/volume*op.elementwise_integral(dcoll, field)

    # This will not make the limiter conservative but it is better than having
    # negative species. This should only be necessary for coarse grids or
    # underresolved regions... If it is knowingly underresolved, then I think we can
    # abstain to ensure "exact" conservation.
    # Most importantly, without this, the limiter doesn't work <smile face>..
    cell_avgs = actx.np.where(actx.np.greater(cell_avgs,0.0), cell_avgs, 0.0)

    # Compute nodal and elementwise max/mins of the field
    mmin_i = op.elementwise_min(dcoll, field)

    mmin = 0.0

    _theta = actx.np.minimum(1., (mmin-cell_avgs)/(mmin_i-cell_avgs-1e-13) )

    return _theta*(field - cell_avgs) + cell_avgs
