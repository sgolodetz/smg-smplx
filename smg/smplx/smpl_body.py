import cv2
import numpy as np
import os
import smplx
import smplx.utils
import torch

from OpenGL.GL import *
from scipy.spatial.transform import Rotation
from typing import Optional, Tuple

from smg.opengl import OpenGLMatrixContext, OpenGLTexture, OpenGLTextureContext, OpenGLUtil
from smg.skeletons import Keypoint, Skeleton3D
from smg.utility import ImageUtil


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
    """An SMPL body."""

    # CONSTRUCTOR

    def __init__(self, gender: str, *, model_dir: Optional[str] = None,
                 texture_coords_filename: Optional[str] = None,
                 texture_image_filename: Optional[str] = None):
        """
        Construct an SMPL body.

        :param gender:                  The gender of the SMPL body model to load.
        :param model_dir:               The directory containing the SMPL body models.
        :param texture_coords_filename: The name of the file containing the UV coordinates for the texture (optional).
        :param texture_image_filename:  The name of the file containing the image for the texture (optional).
        """
        # Try to determine the model directory.
        if model_dir is None:
            model_dir = os.environ.get("SMGLIB_SMPLX_MODEL_DIR")
            if model_dir is None:
                raise RuntimeError(
                    "Could not determine SMPL-X model directory: please add SMGLIB_SMPLX_MODEL_DIR to the environment"
                )

        # Load the SMPL body model.
        # noinspection PyTypeChecker
        self.__model: smplx.SMPL = smplx.create(model_dir, "smpl", gender=gender)

        # Set up the internal arrays.
        # 0 = height (-ve short, +ve tall), 1 = weight (-ve fat, +ve thin),
        # 2 = torso stretch (-ve compress, +ve stretch), etc.
        self.__betas: np.ndarray = np.zeros(self.__model.num_betas, dtype=np.float32)

        # 0:3 = left hip, 3:6 = right hip, etc. (see enumeration values above)
        self.__body_pose: np.ndarray = np.zeros(self.__model.NUM_BODY_JOINTS * 3, dtype=np.float32)

        # 0 = pelvis, 1 = left hip, etc. (see enumeration values above)
        self.__joints: Optional[np.ndarray] = None

        self.__faces: np.ndarray = self.__model.faces
        self.__global_pose: np.ndarray = np.eye(4)
        self.__neutral_joints: Optional[np.ndarray] = None
        self.__vertices: Optional[np.ndarray] = None

        # Load in any texture image that has been specified, along with its UV coordinates.
        self.__texture: Optional[OpenGLTexture] = None
        self.__texture_coords: Optional[np.ndarray] = None
        self.__texture_image: Optional[np.ndarray] = None

        if texture_coords_filename is not None and texture_image_filename is not None:
            self.__texture = OpenGLTexture()
            self.__texture_coords = np.load(texture_coords_filename)
            self.__texture_image = ImageUtil.flip_channels(np.flip(cv2.imread(texture_image_filename), axis=0))

    # PROPERTIES

    @property
    def betas(self) -> np.ndarray:
        """
        Get a copy of the array that contains the shape parameters for the body.

        :return:    A copy of the array that contains the shape parameters for the body.
        """
        return self.__betas.copy()

    @property
    def body_pose(self) -> np.ndarray:
        """
        Get a copy of the array that contains the local rotations for the body's joints.

        :return:    A copy of the array that contains the local rotations for the body's joints.
        """
        return self.__body_pose.copy()

    @property
    def num_betas(self) -> np.ndarray:
        """
        Get the number of shape parameters that the body has.

        :return:    The number of shape parameters that the body has.
        """
        return self.__model.num_betas

    # PUBLIC METHODS

    def render(self, *, colour: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Render the body using OpenGL.

        .. note::
            The implementation is an adapted version of:
            https://sites.google.com/site/dlampetest/python/calculating-normals-of-a-triangle-mesh-using-numpy

        :param colour:  An optional colour with which to render the body if no texture is being used.
        """
        # ~~~
        # Step 1: Prepare the mesh for rendering.
        # ~~~

        # If the mesh doesn't exist yet, early out.
        if self.__vertices is None:
            return

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
            glPushClientAttrib(GL_CLIENT_VERTEX_ARRAY_BIT)

            glEnableClientState(GL_VERTEX_ARRAY)
            glEnableClientState(GL_NORMAL_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, face_vertices)
            glNormalPointer(GL_FLOAT, 0, face_vertex_normals)

            if self.__texture is not None:
                # Make an F*3*2 array that explicitly lists the texture coordinates for each face (F faces,
                # 3 vertices per face, 2 components per UV texture coordinate pair).
                face_tex_coords: np.ndarray = self.__texture_coords[self.__faces]

                with OpenGLTextureContext(self.__texture):
                    self.__texture.set_image(self.__texture_image)
                    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
                    glTexCoordPointer(2, GL_DOUBLE, 0, face_tex_coords)
                    glColor3f(1.0, 1.0, 1.0)
                    glDrawArrays(GL_TRIANGLES, 0, len(face_vertices) * 3)
            else:
                if colour is None:
                    colour = (0.7, 0.7, 0.7)

                glColor3f(*colour)
                glDrawArrays(GL_TRIANGLES, 0, len(face_vertices) * 3)

            glPopClientAttrib()

    def render_from_skeleton(self, skeleton: Skeleton3D, *, colour: Optional[Tuple[float, float, float]] = None,
                             fit_shape: bool = True) -> None:
        """
        Set the pose of the body based on the specified skeleton and then render the body.

        :param skeleton:    The skeleton upon which to base the pose of the body.
        :param colour:      An optional colour with which to render the body if no texture is being used.
        :param fit_shape:   Whether or not to try to fit the body model's shape parameters based on the skeleton.
        """
        self.set_pose_from_skeleton(skeleton, fit_shape=fit_shape)
        self.render(colour=colour)

    def render_joints(self) -> None:
        """Render the body's joints (e.g. for debugging purposes)."""
        if self.__joints is None:
            return

        with OpenGLMatrixContext(GL_MODELVIEW, lambda: OpenGLUtil.mult_matrix(self.__global_pose)):
            glColor3f(1.0, 0.0, 1.0)
            for i in range(24):
                OpenGLUtil.render_sphere(self.__joints[i], 0.02, slices=10, stacks=10)

    def set_manual_pose(self, body_pose: np.ndarray, world_from_midhip: np.ndarray,
                        *, betas: Optional[np.ndarray] = None) -> None:
        """
        Set a manual pose for the body.

        :param body_pose:           An array containing the local rotations for the body's joints.
        :param world_from_midhip:   The global pose of the body's mid-hip joint.
        :param betas:               An array containing the shape parameters for the body (optional).
        """
        # Update the internal array containing the local rotations for the body's joints.
        np.copyto(self.__body_pose, body_pose)

        # Update the internal array containing the body's shape parameters, if new ones have been specified.
        if betas is not None:
            np.copyto(self.__betas, betas)

        # Run the body model to update the global joint positions and mesh vertices.
        self.__joints, self.__vertices = self.__run_model()

        # Calculate the global pose for the body.
        self.__calculate_global_pose(world_from_midhip)

    def set_pose_from_skeleton(self, skeleton: Skeleton3D, *, fit_shape: bool = True) -> None:
        """
        Set the pose of the body based on the specified skeleton.

        :param skeleton:    The skeleton upon which to base the pose of the body.
        :param fit_shape:   Whether or not to try to fit the body model's shape parameters based on the skeleton.
        """
        # Try to get the global pose of the skeleton's mid-hip joint. If this isn't possible, early out.
        world_from_midhip: Optional[np.ndarray] = skeleton.global_keypoint_poses.get("MidHip")
        if world_from_midhip is None:
            return

        # Try to apply the local rotations from the skeleton's keypoints to the joints of the SMPL body.
        self.__try_apply_local_keypoint_rotation(skeleton, "LElbow", SMPLJ_LEFT_ELBOW)
        self.__try_apply_local_keypoint_rotation(skeleton, "LHip", SMPLJ_LEFT_HIP)
        self.__try_apply_local_keypoint_rotation(skeleton, "LKnee", SMPLJ_LEFT_KNEE)
        self.__try_apply_local_keypoint_rotation(skeleton, "LShoulder", SMPLJ_LEFT_SHOULDER)
        self.__try_apply_local_keypoint_rotation(skeleton, "Neck", SMPLJ_NECK)
        self.__try_apply_local_keypoint_rotation(skeleton, "RElbow", SMPLJ_RIGHT_ELBOW)
        self.__try_apply_local_keypoint_rotation(skeleton, "RHip", SMPLJ_RIGHT_HIP)
        self.__try_apply_local_keypoint_rotation(skeleton, "RKnee", SMPLJ_RIGHT_KNEE)
        self.__try_apply_local_keypoint_rotation(skeleton, "RShoulder", SMPLJ_RIGHT_SHOULDER)

        # If shape fitting is enabled:
        if fit_shape:
            # If we haven't yet calculated the positions of the joints when the SMPL body is in its neutral pose,
            # calculate them now.
            if self.__neutral_joints is None:
                self.__betas.fill(0.0)
                self.__neutral_joints, _ = self.__run_model(return_verts=False)

            # Try to stretch the torso of the SMPL body to better fit the detected skeleton.
            midhip_keypoint: Optional[Keypoint] = skeleton.keypoints.get("MidHip")
            neck_keypoint: Optional[Keypoint] = skeleton.keypoints.get("Neck")

            if midhip_keypoint is not None and neck_keypoint is not None:
                midhip_smplj: np.ndarray = SMPLBody.__calculate_midhip_position(self.__neutral_joints)
                neck_smplj: np.ndarray = \
                    (self.__neutral_joints[SMPLJ_LEFT_SHOULDER] + self.__neutral_joints[SMPLJ_RIGHT_SHOULDER]) / 2

                skeleton_torso_length: float = np.linalg.norm(midhip_keypoint.position - neck_keypoint.position)
                neutral_body_torso_length: float = np.linalg.norm(midhip_smplj - neck_smplj)

                # Note: The factor of 100 here was empirically determined but seems to work ok.
                self.__betas[2] = 100 * (skeleton_torso_length - neutral_body_torso_length)

        # Run the body model to update the global joint positions and mesh vertices.
        self.__joints, self.__vertices = self.__run_model()

        # Calculate the global pose for the body.
        self.__calculate_global_pose(world_from_midhip)

    # PRIVATE METHODS

    def __calculate_global_pose(self, world_from_midhip: np.ndarray) -> None:
        """
        Calculate the global pose for the body.

        :param world_from_midhip:   The global pose of the skeleton's mid-hip joint.
        :return:                    The global pose for the body.
        """
        self.__global_pose = world_from_midhip.copy()
        midhip_smplj: np.ndarray = SMPLBody.__calculate_midhip_position(self.__joints)
        self.__global_pose[0:3, 3] -= self.__global_pose[0:3, 0:3] @ midhip_smplj

    def __run_model(self, *, return_verts: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the SMPL body model.

        :param return_verts:    Whether or not to ask the body model to return the mesh vertices.
        :return:                The global joint positions and (if requested) the mesh vertices produced by the model.
        """
        # Run the body model.
        output: smplx.utils.SMPLOutput = self.__model(
            betas=torch.from_numpy(self.__betas).unsqueeze(dim=0),
            body_pose=torch.from_numpy(self.__body_pose).unsqueeze(dim=0),
            return_verts=return_verts
        )

        # Get the global joint positions.
        joints: np.ndarray = output.joints.detach().cpu().numpy().squeeze()

        # If available, also get the mesh vertices.
        vertices: Optional[np.ndarray] = output.vertices.detach().cpu().numpy().squeeze() if return_verts else None

        return joints, vertices

    def __try_apply_local_keypoint_rotation(self, skeleton: Skeleton3D, keypoint_name: str, joint_id: int) -> None:
        """
        Try to apply the specified local keypoint rotation from a skeleton to the specified joint in the
        SMPL body model.

        .. note::
            If the local keypoint rotation can't be provided by the skeleton, for whatever reason, this is a no-op.

        :param skeleton:        The skeleton.
        :param keypoint_name:   The name of the keypoint in the skeleton whose local rotation we want to use.
        :param joint_id:        The ID of the joint in the SMPL body model whose local rotation we want to set.
        """
        local_keypoint_rotation: Optional[np.ndarray] = skeleton.local_keypoint_rotations.get(keypoint_name)
        if local_keypoint_rotation is not None:
            self.__body_pose[(joint_id - 1) * 3:joint_id * 3] = \
                Rotation.from_matrix(local_keypoint_rotation).as_rotvec()

    # PRIVATE STATIC METHODS

    @staticmethod
    def __calculate_midhip_position(joints: np.ndarray) -> np.ndarray:
        """
        Calculate the mid-hip position for the SMPL body, given the known positions of its joints (in some pose).

        :param joints:  The positions of the SMPL body's joints (in some pose).
        :return:        The mid-hip position for the SMPL body.
        """
        return (joints[SMPLJ_PELVIS] + joints[SMPLJ_LEFT_HIP] + joints[SMPLJ_RIGHT_HIP]) / 3

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
