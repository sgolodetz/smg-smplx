import numpy as np
import smplx
import smplx.utils
import torch

from OpenGL.GL import *
from typing import Optional

from smg.skeletons import Skeleton


class SMPLBody:
    """TODO"""

    # CONSTRUCTOR

    def __init__(self, gender: str, *, model_folder: str = "D:/smplx/models"):
        # noinspection PyTypeChecker
        self.__model: smplx.SMPL = smplx.create(model_folder, "smpl", gender=gender)

        self.__body_pose: np.ndarray = np.zeros(self.__model.NUM_BODY_JOINTS * 3, dtype=np.float32)
        self.__faces: np.ndarray = self.__model.faces
        self.__vertices: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def render(self) -> None:
        ###
        glColor3f(0.7, 0.7, 0.7)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)

        # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
        norm = np.zeros(self.__vertices.shape, dtype=self.__vertices.dtype)
        # Create an indexed view into the vertex array using the array of three indices for triangles
        tris = self.__vertices[self.__faces]
        # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
        n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
        # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
        # we need to normalize these, so that our next step weights each normal equally.
        SMPLBody.__normalize_v3(n)
        # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
        # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
        # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
        # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
        norm[self.__faces[:, 0]] += n
        norm[self.__faces[:, 1]] += n
        norm[self.__faces[:, 2]] += n
        SMPLBody.__normalize_v3(norm)

        # To render without the index list, we create a flattened array where
        # the triangle indices are replaced with the actual vertices.

        # first we create a single column index array
        tri_index = self.__faces.reshape((-1))
        # then we create an indexed view into our vertices and normals
        va = self.__vertices[self.__faces]
        no = norm[self.__faces]

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, va)
        glNormalPointer(GL_FLOAT, 0, no)
        glDrawArrays(GL_TRIANGLES, 0, len(va) * 3)
        ###

    def set_from_skeleton(self, skeleton: Skeleton) -> None:
        output: smplx.utils.SMPLOutput = self.__model(
            betas=None,
            body_pose=torch.from_numpy(self.__body_pose).unsqueeze(dim=0),
            return_verts=True
        )
        self.__vertices = output.vertices.detach().cpu().numpy().squeeze()

    # PRIVATE STATIC METHODS

    # See: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
    @staticmethod
    def __normalize_v3(arr):
        ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
        lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
        arr[:,0] /= lens
        arr[:,1] /= lens
        arr[:,2] /= lens
        return arr
