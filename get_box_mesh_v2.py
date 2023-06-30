def get_box_mesh(dim,a,b,n, t=None, periodic=None):
    if periodic is None:
        periodic = (False,)*dim
    if np.isscalar(a):
        a = (a,)*dim
    if np.isscalar(b):
        b = (b,)*dim
    if np.isscalar(n):
        n = (n,)*dim

    dim_names = ["x","y","z"]
    bttf = {}
    for i in range(dim):
        bttf["-"+str(i+1)] = ["-"+dim_names[i]]
        bttf["+"+str(i+1)] = ["+"+dim_names[i]]
    from meshmode.mesh.generation import generate_regular_rect_mesh as gen
    return gen(a=a, b=b, n=n, boundary_tag_to_face=bttf, mesh_type=t,periodic=periodic)
