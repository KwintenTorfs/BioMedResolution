from Assistant_Functions.ImageImport import ImageStack
from Meshing.pvMesh import Mesh

path = r'F:\Project - MTF Validation\5 Identical Acquisitions\KULQCCT_LM_MTF_Reprod_Unknown_20240123_140423643'

stack = ImageStack(path)
print('Stack Complete')

mesh = Mesh(stack)
print('Mesh Complete')

# %%



radials = mesh.select_radial

wed = stack.WED_avg
wed_sd = stack.WED_std
