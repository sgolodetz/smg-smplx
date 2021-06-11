import numpy as np
import smplx
import smplx.utils
import torch

from OpenGL.GL import *
from scipy.spatial.transform import Rotation
from typing import Optional

from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.skeletons import Skeleton


# HELPER ENUMERATIONS

class ESMPLJoint(int):
    """An SMPL joint identifier."""

    # SPECIAL METHODS

    def __str__(self) -> str:
        """
        Get an informal string representation of an SMPL joint identifier.

        :return:    An informal string representation of an SMPL joint identifier.
        """
        if self == SMPLJ_PELVIS:
            return "SMPLJ_PELVIS"
        elif self == SMPLJ_LEFT_HIP:
            return "SMPLJ_LEFT_HIP"
        elif self == SMPLJ_RIGHT_HIP:
            return "SMPLJ_RIGHT_HIP"
        elif self == SMPLJ_SPINE1:
            return "SMPLJ_SPINE1"
        elif self == SMPLJ_LEFT_KNEE:
            return "SMPLJ_LEFT_KNEE"
        elif self == SMPLJ_RIGHT_KNEE:
            return "SMPLJ_RIGHT_KNEE"
        elif self == SMPLJ_SPINE2:
            return "SMPLJ_SPINE2"
        elif self == SMPLJ_LEFT_ANKLE:
            return "SMPLJ_LEFT_ANKLE"
        elif self == SMPLJ_RIGHT_ANKLE:
            return "SMPLJ_RIGHT_ANKLE"
        elif self == SMPLJ_SPINE3:
            return "SMPLJ_SPINE3"
        elif self == SMPLJ_LEFT_FOOT:
            return "SMPLJ_LEFT_FOOT"
        elif self == SMPLJ_RIGHT_FOOT:
            return "SMPLJ_RIGHT_FOOT"
        elif self == SMPLJ_NECK:
            return "SMPLJ_NECK"
        elif self == SMPLJ_LEFT_COLLAR:
            return "SMPLJ_LEFT_COLLAR"
        elif self == SMPLJ_RIGHT_COLLAR:
            return "SMPLJ_RIGHT_COLLAR"
        elif self == SMPLJ_HEAD:
            return "SMPLJ_HEAD"
        elif self == SMPLJ_LEFT_SHOULDER:
            return "SMPLJ_LEFT_SHOULDER"
        elif self == SMPLJ_RIGHT_SHOULDER:
            return "SMPLJ_RIGHT_SHOULDER"
        elif self == SMPLJ_LEFT_ELBOW:
            return "SMPLJ_LEFT_ELBOW"
        elif self == SMPLJ_RIGHT_ELBOW:
            return "SMPLJ_RIGHT_ELBOW"
        elif self == SMPLJ_LEFT_WRIST:
            return "SMPLJ_LEFT_WRIST"
        elif self == SMPLJ_RIGHT_WRIST:
            return "SMPLJ_RIGHT_WRIST"
        elif self == SMPLJ_LEFT_HAND:
            return "SMPLJ_LEFT_HAND"
        elif self == SMPLJ_RIGHT_HAND:
            return "SMPLJ_RIGHT_HAND"
        else:
            return "SMPLJ_UNKNOWN"


SMPLJ_PELVIS = ESMPLJoint(0)
SMPLJ_LEFT_HIP = ESMPLJoint(1)
SMPLJ_RIGHT_HIP = ESMPLJoint(2)
SMPLJ_SPINE1 = ESMPLJoint(3)
SMPLJ_LEFT_KNEE = ESMPLJoint(4)
SMPLJ_RIGHT_KNEE = ESMPLJoint(5)
SMPLJ_SPINE2 = ESMPLJoint(6)
SMPLJ_LEFT_ANKLE = ESMPLJoint(7)
SMPLJ_RIGHT_ANKLE = ESMPLJoint(8)
SMPLJ_SPINE3 = ESMPLJoint(9)
SMPLJ_LEFT_FOOT = ESMPLJoint(10)
SMPLJ_RIGHT_FOOT = ESMPLJoint(11)
SMPLJ_NECK = ESMPLJoint(12)
SMPLJ_LEFT_COLLAR = ESMPLJoint(13)
SMPLJ_RIGHT_COLLAR = ESMPLJoint(14)
SMPLJ_HEAD = ESMPLJoint(15)
SMPLJ_LEFT_SHOULDER = ESMPLJoint(16)
SMPLJ_RIGHT_SHOULDER = ESMPLJoint(17)
SMPLJ_LEFT_ELBOW = ESMPLJoint(18)
SMPLJ_RIGHT_ELBOW = ESMPLJoint(19)
SMPLJ_LEFT_WRIST = ESMPLJoint(20)
SMPLJ_RIGHT_WRIST = ESMPLJoint(21)
SMPLJ_LEFT_HAND = ESMPLJoint(22)
SMPLJ_RIGHT_HAND = ESMPLJoint(23)


# MAIN CLASS

class SMPLBody:
    """An SMPL body model."""

    # CONSTRUCTOR

    def __init__(self, gender: str, *, model_folder: str = "D:/smplx/models"):
        # noinspection PyTypeChecker
        self.__model: smplx.SMPL = smplx.create(model_folder, "smpl", gender=gender)

        # 0:3 = left hip, 3:6 = right hip, etc. (see enumeration values above)
        self.__body_pose: np.ndarray = np.zeros(self.__model.NUM_BODY_JOINTS * 3, dtype=np.float32)

        # 0 = pelvis, 1 = left hip, etc. (see enumeration values above)
        self.__joints: Optional[np.ndarray] = None

        self.__faces: np.ndarray = self.__model.faces
        self.__global_pose: np.ndarray = np.eye(4)
        self.__vertices: Optional[np.ndarray] = None

    # PUBLIC METHODS

    def render(self) -> None:
        """
        Render the body using OpenGL.

        .. note::
            The implementation is an adapted version of:
            https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy
        """
        # ~~~
        # Step 1: Prepare the mesh for rendering.
        # ~~~

        # Make an F*3*3 array that explicitly lists the vertices for each face (F faces, 3 vertices per face,
        # 3 components per vertex).
        face_vertices: np.ndarray = self.__vertices[self.__faces]

        # Compute the normal for each face. The resulting array has shape F*3 (F faces, 3 components per normal).
        face_normals: np.ndarray = np.cross(
            face_vertices[::, 1] - face_vertices[::, 0],
            face_vertices[::, 2] - face_vertices[::, 0]
        )
        SMPLBody.__normalise_inplace(face_normals)

        # Compute the normal for each vertex by averaging the normals of all the faces that contain it.
        # The resulting array has shape V*3 (V vertices, 3 components per normal).
        vertex_normals: np.ndarray = np.zeros_like(self.__vertices)
        vertex_normals[self.__faces[:, 0]] += face_normals
        vertex_normals[self.__faces[:, 1]] += face_normals
        vertex_normals[self.__faces[:, 2]] += face_normals
        SMPLBody.__normalise_inplace(vertex_normals)

        # Make an F*3*3 array that explicitly lists the vertex normals for each face (F faces, 3 vertices per face,
        # 3 components per normal).
        face_vertex_normals: np.ndarray = vertex_normals[self.__faces]

        # ~~~
        # Step 2: Render the mesh.
        # ~~~
        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(self.__global_pose)):
            glColor3f(0.7, 0.7, 0.7)

            glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)
            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, face_vertices)
            glNormalPointer(GL_FLOAT, 0, face_vertex_normals)

            glDrawArrays(GL_TRIANGLES, 0, len(face_vertices) * 3)

            glPopClientAttrib()

    def render_from_skeleton(self, skeleton: Skeleton) -> None:
        """
        Set the pose of the body based on the specified skeleton and then render the body.

        :param skeleton:    The skeleton upon which to base the pose of the body.
        """
        self.set_from_skeleton(skeleton)
        self.render()

    def render_joints(self) -> None:
        """Render the body's joints (e.g. for debugging purposes)."""
        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(self.__global_pose)):
            glColor3f(1.0, 0.0, 1.0)
            for i in range(24):
                OpenGLUtil.render_sphere(self.__joints[i], 0.02, slices=10, stacks=10)

    def set_from_skeleton(self, skeleton: Skeleton) -> None:
        """
        Set the pose of the body based on the specified skeleton.

        :param skeleton:    The skeleton upon which to base the pose of the body.
        """
        # Try to get the global pose of the skeleton's MidHip joint. If this isn't possible, early out.
        midhip_w_t_c: Optional[np.ndarray] = skeleton.global_keypoint_poses.get("MidHip")
        if midhip_w_t_c is None:
            return

        # Apply the local rotations from the skeleton's keypoint to the joints of the SMPL body.
        self.__apply_local_keypoint_rotation(skeleton, "LElbow", SMPLJ_LEFT_ELBOW)
        self.__apply_local_keypoint_rotation(skeleton, "LHip", SMPLJ_LEFT_HIP)
        self.__apply_local_keypoint_rotation(skeleton, "LKnee", SMPLJ_LEFT_KNEE)
        self.__apply_local_keypoint_rotation(skeleton, "LShoulder", SMPLJ_LEFT_SHOULDER)
        self.__apply_local_keypoint_rotation(skeleton, "Neck", SMPLJ_NECK)
        self.__apply_local_keypoint_rotation(skeleton, "RElbow", SMPLJ_RIGHT_ELBOW)
        self.__apply_local_keypoint_rotation(skeleton, "RHip", SMPLJ_RIGHT_HIP)
        self.__apply_local_keypoint_rotation(skeleton, "RKnee", SMPLJ_RIGHT_KNEE)
        self.__apply_local_keypoint_rotation(skeleton, "RShoulder", SMPLJ_RIGHT_SHOULDER)

        # Run the body model to update the mesh and the global joint positions.
        output: smplx.utils.SMPLOutput = self.__model(
            betas=None,
            body_pose=torch.from_numpy(self.__body_pose).unsqueeze(dim=0),
            return_verts=True
        )

        # Get the updated mesh vertices and global joint positions.
        self.__vertices = output.vertices.detach().cpu().numpy().squeeze()
        self.__joints = output.joints.detach().cpu().numpy().squeeze()

        # Calculate a global pose for the body.
        self.__global_pose = midhip_w_t_c.copy()
        midhip_smplj: np.ndarray = \
            (self.__joints[SMPLJ_PELVIS] + self.__joints[SMPLJ_LEFT_HIP] + self.__joints[SMPLJ_RIGHT_HIP]) / 3
        self.__global_pose[0:3, 3] += midhip_smplj

    # PRIVATE METHODS

    def __apply_local_keypoint_rotation(self, skeleton: Skeleton, keypoint_name: str, joint_id: int) -> None:
        """
        Apply the specified local keypoint rotation from a skeleton to the specified joint in the SMPL body model.

        :param skeleton:        The skeleton.
        :param keypoint_name:   The name of the keypoint in the skeleton whose local rotation we want to use.
        :param joint_id:        The ID of the joint in the SMPL body model whose local rotation we want to set.
        """
        self.__body_pose[(joint_id - 1) * 3:joint_id * 3] = \
            Rotation.from_matrix(skeleton.local_keypoint_rotations[keypoint_name]).as_rotvec()

    # PRIVATE STATIC METHODS

    @staticmethod
    def __normalise_inplace(vecs: np.ndarray) -> None:
        """
        Efficiently normalise each of the n vectors in an n*3 array (in-place).

        .. note::
            Every vector in the array must have a non-zero length.

        :param vecs:    The vectors to normalise.
        """
        # See: https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy.
        lens: np.ndarray = np.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2 + vecs[:, 2] ** 2)
        vecs[:, 0] /= lens
        vecs[:, 1] /= lens
        vecs[:, 2] /= lens
