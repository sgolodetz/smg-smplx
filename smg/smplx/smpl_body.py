import math
import numpy as np
import smplx
import smplx.utils
import torch

from OpenGL.GL import *
from scipy.spatial.transform import Rotation
from typing import Dict, Optional

from smg.opengl import OpenGLMatrixContext, OpenGLUtil
from smg.skeletons import Skeleton


# HELPER ENUMERATIONS

class ESMPLJoint(int):
    """An SMPL joint identifier."""

    # SPECIAL METHODS

    def __str__(self) -> str:
        """
        Get a string representation of an SMPL joint identifier.

        :return:    A string representation of an SMPL joint identifier.
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
        elif self == SMPLJ_RIGHT_ANKLE:
            return "SMPLJ_RIGHT_ANKLE"
        elif self == SMPLJ_LEFT_ANKLE:
            return "SMPLJ_LEFT_ANKLE"
        elif self == SMPLJ_SPINE3:
            return "SMPLJ_SPINE3"
        elif self == SMPLJ_RIGHT_FOOT:
            return "SMPLJ_RIGHT_FOOT"
        elif self == SMPLJ_LEFT_FOOT:
            return "SMPLJ_LEFT_FOOT"
        elif self == SMPLJ_NECK:
            return "SMPLJ_NECK"
        elif self == SMPLJ_RIGHT_COLLAR:
            return "SMPLJ_RIGHT_COLLAR"
        elif self == SMPLJ_LEFT_COLLAR:
            return "SMPLJ_LEFT_COLLAR"
        elif self == SMPLJ_HEAD:
            return "SMPLJ_HEAD"
        elif self == SMPLJ_RIGHT_SHOULDER:
            return "SMPLJ_RIGHT_SHOULDER"
        elif self == SMPLJ_LEFT_SHOULDER:
            return "SMPLJ_LEFT_SHOULDER"
        elif self == SMPLJ_RIGHT_ELBOW:
            return "SMPLJ_RIGHT_ELBOW"
        elif self == SMPLJ_LEFT_ELBOW:
            return "SMPLJ_LEFT_ELBOW"
        elif self == SMPLJ_RIGHT_WRIST:
            return "SMPLJ_RIGHT_WRIST"
        elif self == SMPLJ_LEFT_WRIST:
            return "SMPLJ_LEFT_WRIST"
        elif self == SMPLJ_RIGHT_HAND:
            return "SMPLJ_RIGHT_HAND"
        elif self == SMPLJ_LEFT_HAND:
            return "SMPLJ_LEFT_HAND"
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

        # 0 = left hip, 3 = right hip, 6 = spine1, 9 = left knee, 12 = right knee,
        # 15 = spine2, 18 = right ankle, 21 = left ankle, 24 = spine3, 27 = right foot,
        # 30 = left foot, 33 = neck, 36 = right collar, 39 = left collar, 42 = head,
        # 45 = right shoulder, 48 = left shoulder, 51 = right elbow, 54 = left elbow,
        # 57 = right wrist, 60 = left wrist, 63 = right hand, 66 = left hand
        self.__body_pose: np.ndarray = np.zeros(self.__model.NUM_BODY_JOINTS * 3, dtype=np.float32)

        self.__faces: np.ndarray = self.__model.faces
        self.__global_pose: np.ndarray = np.eye(4)

        # TODO: Update this.
        # 0 = pelvis, 1 = left hip, 2 = right hip, 3 = spine1, 4 = left knee, 5 = right knee,
        # 6 = spine2, 7 = left ankle, 8 = right ankle, 9 = spine3, 10 = left foot,
        # 11 = right foot, 12 = neck, 13 = left collar, 14 = right collar, 15 = head,
        # 16 = left shoulder, 17 = right shoulder, 18 = left elbow, 19 = right elbow,
        # 20 = left wrist, 21 = right wrist, 22 = left hand, 23 = right hand
        self.__joints: Optional[np.ndarray] = None

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
        # Step 2: Render the body.
        # ~~~
        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(self.__global_pose)):
            # Render the mesh.
            glColor3f(0.7, 0.7, 0.7)

            glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, face_vertices)
            glNormalPointer(GL_FLOAT, 0, face_vertex_normals)

            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDrawArrays(GL_TRIANGLES, 0, len(face_vertices) * 3)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            glPopClientAttrib()

            # Render the joints.
            glColor3f(1.0, 0.0, 1.0)

            # for i in range(24):
            #     OpenGLUtil.render_sphere(self.__joints[i], 0.02, slices=10, stacks=10)

    def set_from_skeleton(self, skeleton: Skeleton) -> None:
        midhip_keypoint: Optional[Skeleton.Keypoint] = skeleton.keypoints.get("MidHip")
        neck_keypoint: Optional[Skeleton.Keypoint] = skeleton.keypoints.get("Neck")
        if midhip_keypoint is None or neck_keypoint is None:
            return

        joint_rotations, joint_rel_rotations = skeleton.compute_joint_rotations()

        self.__set_joint_rel_rotation(SMPLJ_LEFT_ELBOW, "RElbow", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_LEFT_HIP, "RHip", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_LEFT_KNEE, "LKnee", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_LEFT_SHOULDER, "RShoulder", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_NECK, "Neck", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_RIGHT_ELBOW, "LElbow", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_RIGHT_HIP, "LHip", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_RIGHT_KNEE, "RKnee", joint_rel_rotations)
        self.__set_joint_rel_rotation(SMPLJ_RIGHT_SHOULDER, "LShoulder", joint_rel_rotations)

        # self.__body_pose[(SMPLJ_LEFT_ELBOW - 1) * 3:SMPLJ_LEFT_ELBOW * 3] = np.array([0, 0, math.pi / 4])
        # self.__body_pose[(SMPLJ_LEFT_SHOULDER - 1) * 3:SMPLJ_LEFT_SHOULDER * 3] = np.array([0, 0, math.pi / 4])
        # self.__body_pose[(SMPLJ_NECK - 1) * 3:SMPLJ_NECK * 3] = np.array([0, 0, math.pi / 4])
        # self.__body_pose[(SMPLJ_RIGHT_SHOULDER - 1) * 3:SMPLJ_RIGHT_SHOULDER * 3] = np.array([0, 0, -math.pi/4])

        output: smplx.utils.SMPLOutput = self.__model(
            betas=None,
            body_pose=torch.from_numpy(self.__body_pose).unsqueeze(dim=0),
            return_verts=True
        )

        self.__joints = output.joints.detach().cpu().numpy().squeeze()
        self.__vertices = output.vertices.detach().cpu().numpy().squeeze()

        self.__global_pose = np.eye(4)
        self.__global_pose = skeleton.keypoint_orienters["MidHip"].w_t_c.copy()

        midhip_smplj: np.ndarray = (self.__joints[SMPLJ_PELVIS] + self.__joints[SMPLJ_LEFT_HIP] + self.__joints[SMPLJ_RIGHT_HIP]) / 3
        self.__global_pose[0:3, 3] += midhip_smplj

        # print(np.linalg.norm(midhip_keypoint.position - neck_keypoint.position))
        # print(np.linalg.norm(self.__joints[SMPLJ_PELVIS] - self.__joints[SMPLJ_NECK]))
        # print("===")

    # PRIVATE METHODS

    def __set_joint_rel_rotation(self, joint_id: int, joint_name: str,
                                 joint_rel_rotations: Dict[str, np.ndarray]) -> None:
        # rot = Rotation.from_matrix(joint_rel_rotations[joint_name]).as_rotvec()
        # import vg
        # print(joint_id, joint_name, vg.normalize(rot), np.linalg.norm(rot))
        rot = Rotation.from_matrix(joint_rel_rotations[joint_name]).as_rotvec()
        self.__body_pose[(joint_id - 1) * 3:joint_id * 3] = rot

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
