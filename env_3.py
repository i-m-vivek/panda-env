# Some part of code is taken from https://github.com/qgallouedec/panda-gym
import os
import numpy as np
import pybullet as p
import pybullet_data
from panda import Panda
from objects import YCBObject, InteractiveObj, RBOObject
from joystick import Joystick


class TeleopEnv:
    def __init__(self, input_source="joystick"):
        """
        Creates a TeleOp Env.

        param: input_source: device to use for controlling the robot. [joystick (default), keyboard]
        """

        # create simulation (GUI)
        self.urdfRootPath = pybullet_data.getDataPath()
        p.connect(p.GUI)
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally

        # set up camera
        self._set_camera()

        # load some scene objects
        p.loadURDF(
            os.path.join(self.urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65]
        )
        p.loadURDF(
            os.path.join(self.urdfRootPath, "table/table.urdf"),
            basePosition=[0.5, 0, -0.65],
        )
        self._bodies_idx = {}
        # cube
        # cube_uid = p.loadURDF("cube.urdf", globalScaling=0.05)
        # p.resetBasePositionAndOrientation(cube_uid, [0.7, 0.2, 0.1], [0, 0, 0, 1])

        # object 
        self.object_size  = np.random.uniform(0.03, 0.05)
        if np.random.random() < 0.5:
            x1, y1 = self.get_random_pos(0.4, -0.15, 0.15), self.get_random_pos(0.3, -0.15, 0.15)
            x2, y2 = self.get_random_pos(0.5, -0.15, 0.15), self.get_random_pos(-0.3, -0.15, 0.15)
        else: 
            x1, y1 = self.get_random_pos(0.5, -0.15, 0.15), self.get_random_pos(-0.3, -0.15, 0.15)
            x2, y2 = self.get_random_pos(0.4, -0.15, 0.15), self.get_random_pos(0.3, -0.15, 0.15)
        self.create_box(
            body_name="object",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=2,
            position=[x1, y1, 0],
            rgba_color=[0.9, 0.1, 0.1, 1],
        )
        self.create_box(
            body_name="target",
            half_extents=[
                self.object_size / 2,
                self.object_size / 2,
                self.object_size / 2,
            ],
            mass=0.0,
            ghost=True,
            position=[x2, y2, 0.0],
            rgba_color=[0.9, 0.1, 0.1, 0.3],
        )

        # load a panda robot
        self.panda = Panda()

        # connect to joystick/keyboard for control
        self.input_source = input_source
        if input_source == "joystick":
            print("Using Joystick Inputs")
            self.joystick = Joystick(scale=0.01)
        elif input_source == "keyboard":
            print("Using Keyboard Inputs")
        else:
            NotImplementedError

    def reset(self):
        self.panda.reset()
        return self.panda.state

    def close(self):
        p.disconnect()

    def get_key_inputs(self):
        keys = p.getKeyboardEvents()
        z1 = 0
        z2 = 0
        z3 = 0
        o1 = 0
        o2 = 0
        o3 = 0

        grasp = 1
        reset = False

        if 65295 in keys:
            z1 = -0.015
        if 65296 in keys:
            z1 = 0.015
        if 65297 in keys:
            z3 = 0.015
        if 65298 in keys:
            z3 = -0.015
        if ord("m") in keys:
            z2 = 0.015
        if ord("n") in keys:
            z2 = -0.015

        if ord("u") in keys:
            o1 = 0.015
        if ord("j") in keys:
            o1 = -0.015
        if ord("h") in keys:
            o2 = 0.015
        if ord("k") in keys:
            o2 = -0.015
        if ord("r") in keys:
            o3 = 0.015
        if ord("t") in keys:
            o3 = -0.015

        if ord("c") in keys:
            if grasp:
                grasp = 0
            else:
                grasp = 1

        if ord("x") in keys:
            reset = True

        dpos = np.array([z1, z2, z3])
        dquat = np.array([o1, o2, o3, 0.0])
        return dict(
            dpos=dpos,
            dquat=dquat,
            grasp=grasp,
            reset=reset,
        )

    def step(self):

        # get current state
        state = self.panda.state

        # read in from joystick or keyboard
        if self.input_source == "joystick":
            input = self.joystick.get_controller_state()
        else:
            input = self.get_key_inputs()

        dpos, dquat, grasp, reset = (
            input["dpos"],
            input["dquat"],
            input["grasp"],
            input["reset"],
        )

        # action in this example is the end-effector velocity
        self.panda.step(dposition=dpos, dquaternion=dquat, grasp_open=grasp)

        # take simulation step
        p.stepSimulation()

        # return next_state, reward, done, info
        next_state = self.panda.state
        reward = 0.0
        done = False
        if reset:
            done = True
        info = {}
        return next_state, reward, done, info

    def render(self):
        (width, height, pxl, depth, segmentation) = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.proj_matrix,
        )
        rgb_array = np.array(pxl, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (self.camera_height, self.camera_width, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _set_camera(self):
        self.camera_width = 256
        self.camera_height = 256
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=30,
            cameraPitch=-60,
            cameraTargetPosition=[0.5, -0.2, 0.0],
        )
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.5, 0, 0],
            distance=1.0,
            yaw=90,
            pitch=-50,
            roll=0,
            upAxisIndex=2,
        )
        self.proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(self.camera_width) / self.camera_height,
            nearVal=0.1,
            farVal=100.0,
        )

    def get_random_pos(self, offset, low, high):
        noise = np.random.uniform(low, high)
        offset += noise
        return offset

    def create_box(
        self,
        body_name,
        half_extents,
        mass,
        position,
        rgba_color,
        specular_color=[0, 0, 0, 0],
        ghost=False,
        friction=None,
    ):
        """Create a box.

        Args:
            body_name (str): The name of the box. Must be unique in the sim.
            half_extents (x, y, z): Half size of the box in meters.
            mass (float): The mass in kg.
            position (x, y, z): The position of the box.
            specular_color (r, g, b): RGB specular color.
            rgba_color (r, g, b, a): RGBA color.
            ghost (bool, optional): Whether the box can collide. Defaults to False.
            friction (float, optionnal): The friction. If None, keep the pybullet default
                value. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        return self._create_geometry(
            body_name,
            geom_type=p.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            friction=friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
        self,
        body_name,
        geom_type,
        mass=0,
        position=(0, 0, 0),
        ghost=False,
        friction=None,
        visual_kwargs={},
        collision_kwargs={},
    ):
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See p.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (x, y, z): The position of the geom. Defaults to (0, 0, 0)
            ghost (bool, optional): Whether the geometry can collide. Defaults
                to False.
            friction (float, optionnal): The friction coef.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = p.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = p.createCollisionShape(
                geom_type, **collision_kwargs
            )
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = p.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if friction is not None:
            p.changeDynamics(
                bodyUniqueId=self._bodies_idx[body_name],
                linkIndex=-1,
                lateralFriction=friction,
            )


# Other objects code gist

# obj1 = YCBObject("003_cracker_box")
# obj1.load()
# p.resetBasePositionAndOrientation(
#     obj1.body_id, [0.7, -0.2, 0.1], [0, 0, 0, 1])

# cylinder
# box_uid = p.loadURDF("cylinder.urdf", [0, 0, 1], useFixedBase=False)
# p.resetBasePositionAndOrientation(
#     box_uid, [0.7, 0.2, 0.1], [0, 0, 0, 1])

# cabinet
# obj1 = InteractiveObj("/home/vkmittal14/WORKSPACE/College Work/MTP Work/code/opensource/panda-env/assets/cabinet2/cabinet_0007.urdf", 0.6)
# obj1.load()
# p.resetBasePositionAndOrientation(obj1.body_id, [0.7, -0.2, 0.1], [0, 0, 1, 0])
